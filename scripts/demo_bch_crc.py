"""
BCH + CRC 编码方案演示

根据文档推荐的方案：
- 16 bit short_id + 16 bit CRC = 32 bit 消息
- BCH 编码生成 ~63-100 bit 码字
- 可纠正 3~8 bit 错误
"""
import numpy as np
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.user_manager import UserIDManagerBCH, HybridUserIDManager, UserIDManager
from utils.bch_codec import BCHCodec, create_bch_for_stegastamp
from utils.crc import CRC16, StegaStampCRC


def demo_bch_crc():
    """演示 BCH + CRC 编码方案"""
    print("=" * 70)
    print("BCH + CRC 用户 ID 编码方案演示")
    print("=" * 70)
    print("\n方案说明:")
    print("  编码: user_id -> short_id -> CRC -> BCH 编码 -> 100 bit")
    print("  解码: 100 bit -> BCH 解码纠错 -> CRC 校验 -> short_id -> user_id")
    print("=" * 70)

    # 创建管理器
    manager = UserIDManagerBCH(
        secret_size=100,
        short_id_bits=16,
        crc_bits=16,
        bch_n=63,
        bch_k=32,
        bch_t=3,
        mapping_file="demo_user_mapping.json"
    )

    print("\n配置参数:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试用户
    user_id = "user_001"
    print(f"\n\n{'='*70}")
    print(f"测试用户: {user_id}")
    print(f"{'='*70}")

    # 编码
    secret = manager.encode(user_id)
    print(f"\n[编码结果]")
    print(f"  原始用户: {user_id}")
    print(f"  Short ID: {manager._user_to_short[user_id]}")
    print(f"  编码长度: {len(secret)} bit")
    print(f"  前 20 bit: {secret[:20].astype(int)}")

    # 解码（无错误）
    print(f"\n[解码测试 - 无错误]")
    success, decoded, detail = manager.decode(secret)
    print(f"  结果: {'成功' if success else '失败'}")
    print(f"  解码用户: {decoded}")
    print(f"  详情: {detail}")

    # 错误恢复测试
    print(f"\n[错误恢复测试]")
    for num_errors in [1, 3, 5, 8, 10, 15, 20]:
        corrupted = secret.copy()
        indices = np.random.choice(100, num_errors, replace=False)
        corrupted[indices] = 1 - corrupted[indices]

        success, decoded, detail = manager.decode(corrupted)
        match = (decoded == user_id)

        status = "✓" if success and match else "✗"
        print(f"  {status} {num_errors:2d} bit 错误: 解码{'成功' if success else '失败'}, "
              f"匹配: {match}")


def demo_compare_modes():
    """对比三种验证模式"""
    print("\n\n" + "=" * 70)
    print("三种验证模式对比")
    print("=" * 70)
    print("\n模式:")
    print("  1. Exact:   精确匹配 (0 bit 容错)")
    print("  2. Hamming: 汉明距离阈值 (最多 5 bit 容错)")
    print("  3. BCH_CRC: BCH 纠错 + CRC 校验 (可纠正 3~8 bit)")
    print("=" * 70)

    hybrid = HybridUserIDManager(secret_size=100, mapping_file="demo_user_mapping.json")
    user_id = "user_test"

    # 编码
    hybrid.mode = 'bch_crc'
    secret = hybrid.encode(user_id)
    print(f"\n测试用户: {user_id}")

    # 测试不同错误数
    test_cases = [
        ("无错误", 0),
        ("1 bit 错误", 1),
        ("3 bit 错误", 3),
        ("5 bit 错误", 5),
        ("8 bit 错误", 8),
        ("10 bit 错误", 10),
    ]

    print("\n" + "-" * 70)
    print(f"{'测试场景':<15} {'Exact':>10} {'Hamming':>10} {'BCH_CRC':>10}")
    print("-" * 70)

    for name, num_errors in test_cases:
        # 添加错误
        corrupted = secret.copy()
        if num_errors > 0:
            indices = np.random.choice(100, num_errors, replace=False)
            corrupted[indices] = 1 - corrupted[indices]

        # 三种模式验证
        results = hybrid.compare_all_modes(corrupted, user_id)

        exact_match = results['exact']['match']
        hamming_match = results['hamming']['match']
        bch_match = results['bch_crc']['match']

        print(f"{name:<15} {('✓' if exact_match else '✗'):>10} "
              f"{('✓' if hamming_match else '✗'):>10} "
              f"{('✓' if bch_match else '✗'):>10}")

    print("-" * 70)


def demo_bch_codec():
    """演示 BCH 编解码器"""
    print("\n\n" + "=" * 70)
    print("BCH 编解码器演示")
    print("=" * 70)

    # 创建 BCH 编解码器
    bch = create_bch_for_stegastamp(
        short_id_bits=16,
        crc_bits=16,
        total_bits=100
    )

    print(f"\nBCH 参数: {bch.get_params()}")

    # 测试消息 (32 bit)
    message = np.random.randint(0, 2, 32).astype(np.uint8)
    print(f"\n原始消息 ({len(message)} bit): {message[:16]}...{message[-16:]}")

    # 编码
    codeword = bch.encode(message)
    print(f"编码后 ({len(codeword)} bit): {codeword[:16]}...{codeword[-16:]}")

    # 无错误解码
    success, recovered = bch.decode(codeword)
    print(f"\n无错误解码: {'成功' if success else '失败'}")
    print(f"  恢复消息: {recovered[:16]}...")
    print(f"  正确: {np.array_equal(recovered, message)}")

    # 错误恢复测试
    print(f"\n错误恢复测试:")
    for t in [1, 2, 3, 4, 5, 8, 10]:
        corrupted = codeword.copy()
        indices = np.random.choice(len(codeword), t, replace=False)
        corrupted[indices] = 1 - corrupted[indices]

        success, recovered = bch.decode(corrupted)
        correct = np.array_equal(recovered, message) if success else False

        status = "✓" if success and correct else ("△" if success else "✗")
        print(f"  {status} {t} bit 错误: 解码{'成功' if success else '失败'}, "
              f"恢复正确: {correct}")


def demo_crc():
    """演示 CRC 校验"""
    print("\n\n" + "=" * 70)
    print("CRC 校验演示")
    print("=" * 70)

    from utils.crc import int_to_bits, bits_to_int

    # StegaStamp CRC
    sscrc = StegaStampCRC(short_id_bits=16, crc_bits=16)

    short_id = 12345
    print(f"\n原始 Short ID: {short_id}")

    # 编码
    encoded = sscrc.encode(short_id)
    print(f"编码后 ({len(encoded)} bit): {encoded[:8]}...{encoded[-8:]}")

    # 解码
    success, decoded_id = sscrc.decode(encoded)
    print(f"\n解码: {'成功' if success else '失败'}")
    print(f"  恢复 ID: {decoded_id}")

    # 添加错误
    print(f"\n错误检测测试:")
    for num_errors in [1, 2, 3, 5]:
        corrupted = encoded.copy()
        indices = np.random.choice(len(corrupted), num_errors, replace=False)
        corrupted[indices] = 1 - corrupted[indices]

        success, decoded_id = sscrc.decode(corrupted)
        print(f"  {num_errors} bit 错误: 检测{'通过' if success else '失败'}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("BCH + CRC 编码方案完整演示")
    print("根据文档: StegaStamp BCH CRC 增强方案")
    print("=" * 70 + "\n")

    demo_bch_crc()
    demo_compare_modes()
    demo_bch_codec()
    demo_crc()

    print("\n\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n核心优势:")
    print("  • 从 '检测到错误就拒绝' 升级为 '先纠正小误码，再做身份确认'")
    print("  • BCH 负责纠错 (3~8 bit)，CRC 负责最终验证")
    print("  • 在 JPEG 压缩、轻度噪声场景下识别率显著提升")
    print("  • 保留 Exact 和 Hamming 作为 baseline 对照")
