"""
scripts/check_gpu_env.py - GPU 环境诊断脚本

全面检测 GPU 运行环境，包括:
- NVIDIA 驱动版本
- CUDA 版本
- PyTorch CUDA 可用性
- GPU 设备信息与显存
- Docker NVIDIA Runtime 检测

如果任何检查失败，输出明确的修复命令。

使用方式:
    python scripts/check_gpu_env.py
    python scripts/check_gpu_env.py --verbose
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple


# ============================================================
# 检查结果枚举
# ============================================================

PASS = "✓"
FAIL = "✗"
WARN = "!"


class DiagnosticResult:
    """单项诊断结果"""

    def __init__(self, name: str, passed: bool, value: str, fix_command: str = "") -> None:
        self.name = name
        self.passed = passed
        self.value = value
        self.fix_command = fix_command

    @property
    def icon(self) -> str:
        return PASS if self.passed else FAIL


# ============================================================
# 各项检测函数
# ============================================================

def check_nvidia_driver() -> DiagnosticResult:
    """检测 NVIDIA 驱动"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip().split("\n")[0].strip()
            return DiagnosticResult("NVIDIA 驱动", True, driver_version)
        else:
            return DiagnosticResult(
                "NVIDIA 驱动", False, "nvidia-smi 执行失败",
                "→ 修复: 请安装或更新 NVIDIA 显卡驱动 https://www.nvidia.com/Download/index.html",
            )
    except FileNotFoundError:
        return DiagnosticResult(
            "NVIDIA 驱动", False, "未找到 nvidia-smi",
            "→ 修复: 请安装 NVIDIA 显卡驱动 https://www.nvidia.com/Download/index.html",
        )
    except Exception as e:
        return DiagnosticResult("NVIDIA 驱动", False, f"检测异常: {e}")


def check_cuda_version() -> DiagnosticResult:
    """检测 CUDA 版本"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        # 从 nvidia-smi 输出中获取 CUDA 版本
        result2 = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result2.stdout.split("\n"):
            if "CUDA Version" in line:
                # 形如: "| NVIDIA-SMI 581.15     Driver Version: 581.15     CUDA Version: 13.0     |"
                parts = line.split("CUDA Version")
                if len(parts) > 1:
                    cuda_ver = parts[1].strip().strip("|").strip().lstrip(":").strip()
                    return DiagnosticResult("CUDA 版本", True, cuda_ver)

        return DiagnosticResult(
            "CUDA 版本", False, "未能从 nvidia-smi 中读取 CUDA 版本",
            "→ 修复: 请安装 CUDA Toolkit https://developer.nvidia.com/cuda-downloads",
        )
    except Exception as e:
        return DiagnosticResult("CUDA 版本", False, f"检测异常: {e}")


def check_pytorch_cuda() -> DiagnosticResult:
    """检测 PyTorch CUDA 可用性"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch_version = torch.__version__
            cuda_ver_torch = torch.version.cuda or "N/A"
            return DiagnosticResult(
                "PyTorch CUDA", True,
                f"PyTorch {torch_version} + CUDA {cuda_ver_torch}",
            )
        else:
            return DiagnosticResult(
                "PyTorch CUDA", False, "torch.cuda.is_available() = False",
                "→ 修复: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
            )
    except ImportError:
        return DiagnosticResult(
            "PyTorch CUDA", False, "PyTorch 未安装",
            "→ 修复: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
        )
    except Exception as e:
        return DiagnosticResult("PyTorch CUDA", False, f"检测异常: {e}")


def check_gpu_device() -> DiagnosticResult:
    """检测 GPU 设备信息与显存"""
    try:
        import torch
        if not torch.cuda.is_available():
            return DiagnosticResult(
                "GPU 设备", False, "CUDA 不可用，无法检测 GPU",
                "→ 修复: 请先解决 PyTorch CUDA 问题",
            )
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        # 兼容不同 PyTorch 版本的属性名
        mem_total = getattr(props, 'total_mem', None) or getattr(props, 'total_memory', None) or 0
        mem_total = mem_total / (1024 ** 3)
        mem_info = f"{mem_total:.1f}GB"
        device_info = f"{device_name} ({mem_info})" if device_count == 1 else \
            f"{device_count}x GPU, 主设备: {device_name} ({mem_info})"
        passed = mem_total >= 4.0  # 至少 4GB 显存
        fix = ""
        if not passed:
            fix = "→ 警告: 显存不足 4GB，建议使用更大显存的 GPU"
        return DiagnosticResult("GPU 设备", passed, device_info, fix)
    except ImportError:
        return DiagnosticResult(
            "GPU 设备", False, "PyTorch 未安装",
            "→ 修复: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
        )
    except Exception as e:
        return DiagnosticResult("GPU 设备", False, f"检测异常: {e}")


def check_docker_nvidia() -> DiagnosticResult:
    """检测 Docker NVIDIA Runtime"""
    # 先检查 Docker 是否安装
    docker_cmd = shutil.which("docker")
    if docker_cmd is None:
        return DiagnosticResult(
            "Docker NVIDIA Runtime", False, "Docker 未安装",
            "→ 修复: 请安装 Docker Desktop https://www.docker.com/products/docker-desktop/",
        )

    try:
        # 检查 Docker 是否运行
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return DiagnosticResult(
                "Docker NVIDIA Runtime", False, "Docker 未运行或无法连接",
                "→ 修复: 请启动 Docker Desktop",
            )

        runtimes_output = result.stdout.strip()
        if "nvidia" in runtimes_output.lower():
            return DiagnosticResult("Docker NVIDIA Runtime", True, "已配置")
        else:
            return DiagnosticResult(
                "Docker NVIDIA Runtime", False,
                f"未检测到 nvidia runtime (当前: {runtimes_output})",
                "→ 修复: 安装 NVIDIA Container Toolkit\n"
                "  Linux: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html\n"
                "  Windows: 确保 Docker Desktop 设置中启用了 GPU 支持",
            )
    except subprocess.TimeoutExpired:
        return DiagnosticResult(
            "Docker NVIDIA Runtime", False, "Docker 命令超时",
            "→ 修复: 请确认 Docker 服务正常运行",
        )
    except Exception as e:
        return DiagnosticResult("Docker NVIDIA Runtime", False, f"检测异常: {e}")


def check_gpu_memory_detail() -> DiagnosticResult:
    """检测 GPU 可用显存（通过 nvidia-smi）"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                total = float(parts[0].strip())
                used = float(parts[1].strip())
                free = float(parts[2].strip())
                return DiagnosticResult(
                    "GPU 显存详情", True,
                    f"总计 {total:.0f}MB / 已用 {used:.0f}MB / 可用 {free:.0f}MB",
                )
        return DiagnosticResult("GPU 显存详情", False, "无法读取显存信息")
    except Exception as e:
        return DiagnosticResult("GPU 显存详情", False, f"检测异常: {e}")


# ============================================================
# 主诊断流程
# ============================================================

def run_diagnostics(verbose: bool = False) -> List[DiagnosticResult]:
    """运行所有诊断检查"""
    results = []

    print("正在检测 GPU 环境...\n")

    # 1. NVIDIA 驱动
    results.append(check_nvidia_driver())

    # 2. CUDA 版本
    results.append(check_cuda_version())

    # 3. PyTorch CUDA
    results.append(check_pytorch_cuda())

    # 4. GPU 设备
    results.append(check_gpu_device())

    # 5. GPU 显存详情（仅 verbose 模式）
    if verbose:
        results.append(check_gpu_memory_detail())

    # 6. Docker NVIDIA Runtime
    results.append(check_docker_nvidia())

    return results


def print_report(results: List[DiagnosticResult]) -> None:
    """打印诊断报告"""
    print("=" * 50)
    print("  GPU 环境诊断报告")
    print("=" * 50)
    print()

    all_passed = True
    for r in results:
        status_icon = r.icon
        print(f"[{status_icon}] {r.name}: {r.value}")
        if not r.passed and r.fix_command:
            for line in r.fix_command.split("\n"):
                print(f"    {line}")
            all_passed = False
        elif not r.passed:
            all_passed = False

    print()
    print("-" * 50)

    if all_passed:
        print("状态: 所有检查通过，GPU 环境就绪")
    else:
        failed_count = sum(1 for r in results if not r.passed)
        print(f"状态: {failed_count} 项检查未通过，请根据上方提示修复")

    print()


def get_system_info() -> None:
    """打印系统基本信息"""
    print(f"系统: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    print()


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="GPU 环境诊断工具")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    args = parser.parse_args()

    print()
    get_system_info()
    results = run_diagnostics(verbose=args.verbose)
    print_report(results)

    # 返回码: 0=全部通过, 1=有失败项
    has_failure = any(not r.passed for r in results)
    sys.exit(1 if has_failure else 0)


if __name__ == "__main__":
    main()
