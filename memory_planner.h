/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MEMORY_PLANNER_MEMORY_PLANNER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MEMORY_PLANNER_MEMORY_PLANNER_H_

#include "error_reporter.h"

namespace tflite {

// Interface class for planning the layout of memory buffers during the execution
// of a graph. 
class MemoryPlanner {
 public:
  MemoryPlanner() {}
  virtual ~MemoryPlanner() {}

  virtual bool AddBuffer(tflite::ErrorReporter* error_reporter, int size, int first_time_used, int last_time_used) = 0;

  virtual int GetMaximumMemorySize() = 0;
  virtual int GetBufferCount() = 0;
  virtual bool GetOffsetForBuffer(tflite::ErrorReporter* error_reporter, int buffer_index, int* offset) = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MEMORY_PLANNER_MEMORY_PLANNER_H_
