//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  

import m5
from m5.objects import Root

from gem5.utils.requires import requires
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.memory import DualChannelDDR4_2400
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.simulate.simulator import Simulator
from gem5.resources.resource import obtain_resource, FileResource
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
  PrivateL1PrivateL2CacheHierarchy,
)

requires(isa_required=ISA.RISCV)

cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
  l1d_size="64kB", l1i_size="64kB", l2_size="1024kB"
)

memory = DualChannelDDR4_2400(size="64GB")

processor = SimpleProcessor(
  cpu_type=CPUTypes.TIMING, isa=ISA.RISCV, num_cores=64
)

board = SimpleBoard(
  clk_freq="2GHz",
  processor=processor,
  memory=memory,
  cache_hierarchy=cache_hierarchy,
)

board.set_se_binary_workload(
  FileResource("/home/da/inference_test/torch_extension_cpp_thread_gemm_without_time"),
  arguments = [] # add arguments here if the binary requires arguments
)

simulator = Simulator(board=board)
simulator.run()
