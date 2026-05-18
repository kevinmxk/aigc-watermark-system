"""
CRC 循环冗余校验模块
支持 CRC-8, CRC-16, CRC-32
用于 StegaStamp 用户 ID 最终验证
"""
import numpy as np
from typing import Optional


class CRCCalculator:
    """
    CRC 计算器

    支持标准 CRC 算法:
    - CRC-8: 多项式 0x07, 初始值 0x00
    - CRC-16 (CCITT): 多项式 0x1021, 初始值 0xFFFF
    - CRC-32: 多项式 0x04C11DB7, 初始值 0xFFFFFFFF
    """

    # CRC 表（预计算以提高速度）
    CRC8_TABLE = None
    CRC16_TABLE = None
    CRC32_TABLE = None

    def __init__(self, width: int = 16, poly: Optional[int] = None,
                 init: Optional[int] = None, xor_out: Optional[int] = None):
        """
        初始化 CRC 计算器

        Args:
            width: CRC 位数 (8, 16, 32)
            poly: CRC 多项式 (可选，使用标准值)
            init: 初始值 (可选)
            xor_out: 输出异或值 (可选)
        """
        self.width = width
        self.mask = (1 << width) - 1

        # 使用标准参数
        if width == 8:
            self.poly = poly or 0x07
            self.init = init if init is not None else 0x00
            self.xor_out = xor_out if xor_out is not None else 0x00
        elif width == 16:
            # CRC-16-CCITT
            self.poly = poly or 0x1021
            self.init = init if init is not None else 0xFFFF
            self.xor_out = xor_out if xor_out is not None else 0x0000
        elif width == 32:
            # CRC-32 (IEEE 802.3)
            self.poly = poly or 0x04C11DB7
            self.init = init if init is not None else 0xFFFFFFFF
            self.xor_out = xor_out if xor_out is not None else 0xFFFFFFFF
        else:
            raise ValueError(f"Unsupported CRC width: {width}")

        # 生成 CRC 表
        self.table = self._generate_table()

    def _generate_table(self) -> list:
        """生成 CRC 查找表"""
        table = []
        for byte in range(256):
            crc = byte << (self.width - 8)
            for _ in range(8):
                if crc & (1 << (self.width - 1)):
                    crc = ((crc << 1) ^ self.poly) & self.mask
                else:
                    crc = (crc << 1) & self.mask
            table.append(crc)
        return table

    def calculate(self, data: bytes) -> int:
        """
        计算数据的 CRC 值

        Args:
            data: 输入字节数据

        Returns:
            CRC 值 (整数)
        """
        crc = self.init

        for byte in data:
            if self.width == 8:
                crc = self.table[(crc ^ byte) & 0xFF]
            elif self.width == 16:
                crc = ((crc << 8) ^ self.table[((crc >> 8) ^ byte) & 0xFF]) & self.mask
            elif self.width == 32:
                crc = ((crc >> 8) ^ self.table[(crc ^ byte) & 0xFF]) & self.mask

        return (crc ^ self.xor_out) & self.mask

    def calculate_bits(self, bits: np.ndarray) -> int:
        """
        计算 bit 数组的 CRC

        Args:
            bits: bit 数组 (numpy array of 0/1)

        Returns:
            CRC 值 (整数)
        """
        # 将 bit 数组转换为字节
        bits = bits.flatten()
        n_bytes = (len(bits) + 7) // 8
        data = bytearray(n_bytes)

        for i, bit in enumerate(bits):
            if bit:
                data[i // 8] |= (1 << (7 - i % 8))

        return self.calculate(bytes(data))

    def verify(self, data: bytes, expected_crc: int) -> bool:
        """
        验证数据的 CRC

        Args:
            data: 数据
            expected_crc: 期望的 CRC 值

        Returns:
            是否匹配
        """
        return self.calculate(data) == expected_crc

    def verify_bits(self, bits: np.ndarray, expected_crc: int) -> bool:
        """
        验证 bit 数组的 CRC

        Args:
            bits: bit 数组
            expected_crc: 期望的 CRC 值

        Returns:
            是否匹配
        """
        return self.calculate_bits(bits) == expected_crc

    def int_to_bits(self, crc: int) -> np.ndarray:
        """
        将 CRC 整数转换为 bit 数组

        Args:
            crc: CRC 值

        Returns:
            bit 数组
        """
        bits = []
        for i in range(self.width):
            bits.append((crc >> (self.width - 1 - i)) & 1)
        return np.array(bits, dtype=np.uint8)


class CRC8(CRCCalculator):
    """CRC-8 计算器"""
    def __init__(self):
        super().__init__(width=8)


class CRC16(CRCCalculator):
    """CRC-16 (CCITT) 计算器"""
    def __init__(self):
        super().__init__(width=16)


class CRC32(CRCCalculator):
    """CRC-32 (IEEE 802.3) 计算器"""
    def __init__(self):
        super().__init__(width=32)


class StegaStampCRC:
    """
    StegaStamp 专用的 CRC 校验类

    针对 short_id + CRC 结构优化
    """

    def __init__(self, short_id_bits: int = 16, crc_bits: int = 16):
        """
        Args:
            short_id_bits: short_id 位数 (默认 16)
            crc_bits: CRC 位数 (默认 16，支持 8)
        """
        self.short_id_bits = short_id_bits
        self.crc_bits = crc_bits

        if crc_bits == 8:
            self.crc_calc = CRC8()
        elif crc_bits == 16:
            self.crc_calc = CRC16()
        else:
            raise ValueError(f"Unsupported CRC bits: {crc_bits}")

    def encode(self, short_id: int) -> np.ndarray:
        """
        为 short_id 计算 CRC 并返回完整 bit 数组

        Args:
            short_id: 用户短 ID (整数)

        Returns:
            short_id_bits + CRC bits
        """
        # short_id 转 bit
        short_bits = int_to_bits(short_id, self.short_id_bits)

        # 计算 CRC
        crc_val = self.crc_calc.calculate_bits(short_bits)
        crc_bits = self.crc_calc.int_to_bits(crc_val)

        # 组合
        return np.concatenate([short_bits, crc_bits])

    def decode(self, bits: np.ndarray) -> tuple:
        """
        解码并验证 CRC

        Args:
            bits: short_id_bits + CRC bits

        Returns:
            (成功标志, short_id)
        """
        if len(bits) < self.short_id_bits + self.crc_bits:
            return False, 0

        # 分离 short_id 和 CRC
        short_bits = bits[:self.short_id_bits]
        received_crc_bits = bits[self.short_id_bits:self.short_id_bits + self.crc_bits]

        # 计算 CRC
        expected_crc = self.crc_calc.calculate_bits(short_bits)
        received_crc = bits_to_int(received_crc_bits)

        # 验证
        if expected_crc != received_crc:
            return False, 0

        # 转换 short_id
        short_id = bits_to_int(short_bits)
        return True, short_id


# 辅助函数
def int_to_bits(value: int, num_bits: int) -> np.ndarray:
    """整数转 bit 数组 (大端序)"""
    bits = []
    for i in range(num_bits):
        bits.append((value >> (num_bits - 1 - i)) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_int(bits: np.ndarray) -> int:
    """bit 数组转整数 (大端序)"""
    bits = bits.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def bytes_to_bits(data: bytes) -> np.ndarray:
    """字节转 bit 数组"""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """bit 数组转字节"""
    bits = bits.flatten()
    n_bytes = (len(bits) + 7) // 8
    result = bytearray(n_bytes)
    for i, bit in enumerate(bits):
        if bit:
            result[i // 8] |= (1 << (7 - i % 8))
    return bytes(result)


def test_crc():
    """测试 CRC"""
    print("=" * 60)
    print("CRC 测试")
    print("=" * 60)

    # 测试 CRC-16
    crc16 = CRC16()
    print("\nCRC-16 测试:")

    data = b"Hello, World!"
    crc_val = crc16.calculate(data)
    print(f"  数据: {data}")
    print(f"  CRC-16: 0x{crc_val:04X} ({crc_val})")

    # 验证
    is_valid = crc16.verify(data, crc_val)
    print(f"  验证: {'通过' if is_valid else '失败'}")

    # 测试 bit 数组
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 0] * 2)
    crc_bit = crc16.calculate_bits(bits)
    print(f"\n  Bit 数组 CRC: 0x{crc_bit:04X}")

    # 测试 bit 转换
    crc_bits = crc16.int_to_bits(crc_bit)
    print(f"  CRC bit 数组: {crc_bits}")

    # 测试 StegaStamp CRC
    print("\n--- StegaStamp CRC 测试 ---")
    sscrc = StegaStampCRC(short_id_bits=16, crc_bits=16)

    short_id = 12345
    encoded_bits = sscrc.encode(short_id)
    print(f"\n  Short ID: {short_id}")
    print(f"  编码后 ({len(encoded_bits)} bit): {encoded_bits[:8]}...{encoded_bits[-8:]}")

    # 解码验证
    success, decoded_id = sscrc.decode(encoded_bits)
    print(f"  解码结果: {'成功' if success else '失败'}, ID={decoded_id}")

    # 添加错误测试
    print("\n--- 错误恢复测试 ---")
    corrupted = encoded_bits.copy()
    corrupted[5] = 1 - corrupted[5]
    print(f"  添加 1 bit 错误")

    success, decoded_id = sscrc.decode(corrupted)
    print(f"  CRC 检测: {'通过' if success else '失败'}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_crc()
