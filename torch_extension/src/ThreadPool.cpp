//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  
#include "ThreadPool.h"

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this]
            {
                this->worker_loop();
            }
        );
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker: workers)
        worker.join();
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock,
                [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty())
                return;
            task = std::move(tasks.front());
            tasks.pop();
        }

        task();
    }
}