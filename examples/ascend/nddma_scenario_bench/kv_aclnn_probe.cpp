#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_pa_kv_cache.h"

#ifdef KV_COMPARE_LOOP
void KvApiLoopLaunch(int keyBytes, int groups, aclrtStream stream, void* kcache, void* vcache,
                     void* table, void* lengths, void* offsets, void* key, void* value);
#endif

static void Check(int status, const char* operation) {
    if (status != 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}

struct Tensor {
    void* allocation = nullptr;
    aclTensor* tensor = nullptr;
    size_t bytes;
    size_t guard;
    Tensor(const std::vector<int64_t>& shape, aclDataType dtype, size_t elementBytes,
           aclFormat format, const char* name, size_t guardBytes = 0)
        : bytes(elementBytes), guard(guardBytes) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i-- > 0;) {
            bytes *= shape[i];
            if (i + 1 < shape.size()) strides[i] = strides[i + 1] * shape[i + 1];
        }
        Check(aclrtMalloc(&allocation, bytes + 2 * guard, ACL_MEM_MALLOC_NORMAL_ONLY), "malloc");
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, format,
                                shape.data(), shape.size(), static_cast<uint8_t*>(allocation) + guard);
        if (!tensor) {
            aclrtFree(allocation);
            allocation = nullptr;
            throw std::runtime_error(std::string("create tensor: ") + name);
        }
        std::cout << "TENSOR," << name << ",dtype=" << dtype << ",format=" << format << ",shape=";
        for (auto x : shape) std::cout << x << ':';
        std::cout << ",stride=";
        for (auto x : strides) std::cout << x << ':';
        std::cout << ",bytes=" << bytes << '\n';
    }
    Tensor(const Tensor&) = delete;
    ~Tensor() {
        if (tensor) aclDestroyTensor(tensor);
        if (allocation) aclrtFree(allocation);
    }
    void Upload(const void* source, size_t count) {
        if (count != bytes) throw std::runtime_error("upload size mismatch");
        Check(aclrtMemcpy(static_cast<uint8_t*>(allocation) + guard, bytes, source, count,
                          ACL_MEMCPY_HOST_TO_DEVICE), "upload");
    }
    void Fill(uint8_t value) {
        std::vector<uint8_t> data(bytes + 2 * guard, value);
        Check(aclrtMemcpy(allocation, data.size(), data.data(), data.size(), ACL_MEMCPY_HOST_TO_DEVICE), "fill");
    }
    void Verify(const std::vector<uint8_t>& expected, const char* name) {
        if (expected.size() != bytes) throw std::runtime_error("expected size mismatch");
        std::vector<uint8_t> actual(bytes + 2 * guard), guarded(bytes + 2 * guard, 0xa5);
        std::copy(expected.begin(), expected.end(), guarded.begin() + guard);
        Check(aclrtMemcpy(actual.data(), actual.size(), allocation, actual.size(), ACL_MEMCPY_DEVICE_TO_HOST), "download");
        auto mismatch = std::mismatch(actual.begin(), actual.end(), guarded.begin());
        if (mismatch.first != actual.end()) {
            throw std::runtime_error(std::string(name) + " byte mismatch at allocation offset " +
                                     std::to_string(mismatch.first - actual.begin()));
        }
    }
};

static std::vector<uint8_t> Pack(const std::vector<uint8_t>& nd, int blocks, int blockSize,
                                 int width, bool nz) {
    if (!nz) return nd;
    std::vector<uint8_t> packed(nd.size());
    for (int p = 0; p < blocks; ++p)
        for (int t = 0; t < blockSize; ++t)
            for (int x = 0; x < width; ++x)
                packed[(static_cast<size_t>(p) * (width / 32) + x / 32) * blockSize * 32 + t * 32 + x % 32] =
                    nd[(static_cast<size_t>(p) * blockSize + t) * width + x];
    return packed;
}

static std::vector<uint8_t> Golden(const std::vector<uint8_t>& nd, const std::vector<int32_t>& table,
                                   const std::vector<int32_t>& lengths, int blockSize, int width) {
    std::vector<uint8_t> result;
    for (size_t b = 0; b < lengths.size(); ++b)
        for (int t = 0; t < lengths[b]; ++t) {
            int page = table[b * 9 + 1 + t / blockSize];
            size_t index = (static_cast<size_t>(page) * blockSize + t % blockSize) * width;
            result.insert(result.end(), nd.begin() + index, nd.begin() + index + width);
        }
    return result;
}

static void Run(int device, bool nz, bool fp8, int groups, bool tail, int repeats) {
    constexpr int blockSize = 128, keyDim = 512, valueDim = 64;
    const int blocks = groups * 9 + 3, keyBytes = fp8 ? 1 : 2;
    const int kw = keyDim * keyBytes, vw = valueDim * 2;
    std::vector<int32_t> table(groups * 9), lengths(groups), offsets(groups, blockSize);
    int tokens = 0;
    for (int b = 0; b < groups; ++b) {
        lengths[b] = tail ? 7 * blockSize + 1 : 8 * blockSize;
        tokens += lengths[b];
        for (int j = 0; j < 9; ++j) table[b * 9 + j] = (b * 5 + j * 7 + 3) % blocks;
    }
    auto cacheShape = [=](int dim, int bytes) -> std::vector<int64_t> {
        if (nz) return {blocks, dim * bytes / 32, blockSize, 32 / bytes};
        return {blocks, blockSize, 1, dim};
    };
    auto outputShape = [=](int dim) -> std::vector<int64_t> {
        if (nz) return {tokens, dim};
        return {tokens, 1, dim};
    };
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create stream");
    try {
        const aclDataType kd = fp8 ? ACL_FLOAT8_E4M3FN : ACL_BF16;
        const aclFormat format = nz ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
        Tensor kcache(cacheShape(keyDim, keyBytes), kd, keyBytes, format, "key_cache");
        Tensor vcache(cacheShape(valueDim, 2), ACL_BF16, 2, format, "value_cache");
        Tensor bt({groups, 9}, ACL_INT32, 4, ACL_FORMAT_ND, "block_tables");
        Tensor sl({groups}, ACL_INT32, 4, ACL_FORMAT_ND, "seq_lens");
        Tensor so({groups}, ACL_INT32, 4, ACL_FORMAT_ND, "seq_offset");
        Tensor ko(outputShape(keyDim), kd, keyBytes, ACL_FORMAT_ND, "key", 64);
        Tensor vo(outputShape(valueDim), ACL_BF16, 2, ACL_FORMAT_ND, "value", 64);
        bt.Upload(table.data(), table.size() * 4);
        sl.Upload(lengths.data(), lengths.size() * 4);
        so.Upload(offsets.data(), offsets.size() * 4);
        std::vector<uint8_t> kn(static_cast<size_t>(blocks) * blockSize * kw);
        std::vector<uint8_t> vn(static_cast<size_t>(blocks) * blockSize * vw);
        char norm[] = "Norm", paNz[] = "PA_NZ";
        auto launch = [&](bool candidate = false) {
#ifdef KV_COMPARE_LOOP
            if (candidate) {
                KvApiLoopLaunch(keyBytes, groups, stream, kcache.allocation, vcache.allocation,
                                bt.allocation, sl.allocation, so.allocation,
                                static_cast<uint8_t*>(ko.allocation) + ko.guard,
                                static_cast<uint8_t*>(vo.allocation) + vo.guard);
                Check(aclrtSynchronizeStream(stream), "candidate synchronize");
                std::cout << "CANDIDATE,LOOP\n";
                return;
            }
#else
            (void)candidate;
#endif
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            Check(aclnnGatherPaKvCacheGetWorkspaceSize(kcache.tensor, vcache.tensor, bt.tensor,
                      sl.tensor, ko.tensor, vo.tensor, so.tensor, nz ? paNz : norm, false,
                      &size, &executor), "gather workspace");
            void* workspace = nullptr;
            if (size) Check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_NORMAL_ONLY), "workspace malloc");
            int status = aclnnGatherPaKvCache(workspace, size, executor, stream);
            int sync = aclrtSynchronizeStream(stream);
            if (workspace) aclrtFree(workspace);
            Check(status, "gather launch");
            Check(sync, "gather synchronize");
            std::cout << "WORKSPACE," << size << '\n';
        };
        for (int pattern = 0; pattern < 3; ++pattern) {
            uint32_t state = 0x5a17u + pattern;
            for (auto* bytes : {&kn, &vn})
                for (size_t i = 0; i < bytes->size(); ++i) {
                    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                    (*bytes)[i] = pattern == 0 ? state & 255 : pattern == 1 ? (i / 32 + i * 17) & 255 : 0x3c;
                }
            auto kp = Pack(kn, blocks, blockSize, kw, nz), vp = Pack(vn, blocks, blockSize, vw, nz);
            kcache.Upload(kp.data(), kp.size());
            vcache.Upload(vp.data(), vp.size());
            ko.Fill(0xa5); vo.Fill(0xa5);
            launch();
            ko.Verify(Golden(kn, table, lengths, blockSize, kw), "key");
            vo.Verify(Golden(vn, table, lengths, blockSize, vw), "value");
            std::cout << "CHECK," << pattern << ",PASS\n";
#ifdef KV_COMPARE_LOOP
            ko.Fill(0xa5); vo.Fill(0xa5);
            launch(true);
            ko.Verify(Golden(kn, table, lengths, blockSize, kw), "candidate key");
            vo.Verify(Golden(vn, table, lengths, blockSize, vw), "candidate value");
            std::cout << "CANDIDATE_CHECK," << pattern << ",PASS\n";
#endif
        }
        for (int i = 0; i < repeats; ++i) {
#ifdef KV_COMPARE_LOOP
            launch(i % 2 == 0);
            launch(i % 2 != 0);
#else
            launch();
#endif
        }
#ifdef KV_COMPARE_LOOP
        for (bool candidate : {false, true}) {
            ko.Fill(0xa5); vo.Fill(0xa5);
            launch(candidate);
            ko.Verify(Golden(kn, table, lengths, blockSize, kw), "post key");
            vo.Verify(Golden(vn, table, lengths, blockSize, vw), "post value");
            std::cout << "POST_CHECK," << candidate << ",PASS\n";
        }
#endif
        ko.Verify(Golden(kn, table, lengths, blockSize, kw), "key final");
        vo.Verify(Golden(vn, table, lengths, blockSize, vw), "value final");
        std::cout << "RESULT,PASS,device=" << device << ",layout=" << (nz ? "PA_NZ" : "Norm")
                  << ",key_dtype=" << (fp8 ? "FP8_E4M3FN" : "BF16") << ",value_dtype=BF16,groups="
                  << groups << ",tail=" << tail
#ifdef KV_COMPARE_LOOP
                  << ",calls_per_mode=" << repeats + 4 << '\n';
#else
                  << ",calls=" << repeats + 3 << '\n';
#endif
    } catch (...) {
        aclrtDestroyStream(stream);
        throw;
    }
    Check(aclrtDestroyStream(stream), "destroy stream");
}

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    if (argc != 7) {
        std::cerr << "Usage: kv_aclnn_probe DEVICE Norm|PA_NZ bf16|fp8 GROUPS TAIL REPEATS\n";
        return 2;
    }
    int device = -1;
    bool initialized = false, selected = false;
    int result = 0;
    try {
        device = std::stoi(argv[1]);
        const std::string layout = argv[2], dtype = argv[3];
        int groups = std::stoi(argv[4]), tail = std::stoi(argv[5]), repeats = std::stoi(argv[6]);
        if ((device != 0 && device != 1) || (layout != "Norm" && layout != "PA_NZ") ||
            (dtype != "bf16" && dtype != "fp8") || groups < 1 || groups > 64 ||
            tail < 0 || tail > 1 || repeats < 0 || repeats > 200)
            throw std::runtime_error("unsupported probe arguments; use documented bounded cases");
#ifdef KV_COMPARE_LOOP
        if (layout != "PA_NZ") throw std::runtime_error("Loop comparison requires PA_NZ; use the API-only probe for Norm");
#endif
        Check(aclInit(nullptr), "acl init"); initialized = true;
        Check(aclrtSetDevice(device), "set device"); selected = true;
        const char* soc = aclrtGetSocName();
        std::cout << "SOC," << (soc ? soc : "unknown") << '\n';
#ifdef KV_COMPARE_LOOP
        if (!soc || std::string(soc) != "Ascend950PR_9579")
            throw std::runtime_error("Matched loop comparison expects Ascend950PR_9579; regenerate core and UB settings for another SoC");
#endif
        Run(device, layout == "PA_NZ", dtype == "fp8", groups, tail, repeats);
    } catch (const std::exception& e) {
        std::cerr << "ERROR," << e.what() << '\n';
        result = 1;
    }
    if (selected) aclrtResetDevice(device);
    if (initialized) aclFinalize();
    return result;
}
