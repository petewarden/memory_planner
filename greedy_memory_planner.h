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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MEMORY_PLANNER_GREEDY_MEMORY_PLANNER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MEMORY_PLANNER_GREEDY_MEMORY_PLANNER_H_

#include "memory_planner.h"

namespace tflite {

// A memory planner that uses a greedy algorithm to arrange buffers in memory
// to minimize the overall arena size needed.
//
// The algorithm works like this:
//  - The client enters the buffer information through AddBuffer().
//  - When a function like GetOffsetForBuffer() is called, the
//    CalculateOffsetsIfNeeded() method is invoked.
//  - If an up to date plan is not already present, one will be calculated.
//  - The buffers are sorted in descending order of size.
//  - The largest buffer is placed at offset zero.
//  - The rest of the buffers are looped through in descending size order.
//  - The other buffers that need to be in memory at the same time are found.
//  - The first gap between active buffers that the current buffer fits into 
//    will be used.
//  - If no large-enough gap is found, the current buffer is placed after the
//    last active buffer.
//  - This continues until all buffers are placed, and the offsets stored.
//
// This is not guaranteed to produce the best placement, since that's an
// NP-Complete problem, but in practice it should produce one that's decent.
class GreedyMemoryPlanner : public MemoryPlanner {
 public:
  GreedyMemoryPlanner();
  virtual ~GreedyMemoryPlanner() override;

  // Record details of a buffer we want to place.
  virtual bool AddBuffer(ErrorReporter* error_reporter, int size, int first_time_used, int last_time_used) override;

  // Returns the high-water mark of used memory. This is the minimum size of a
  // memory arena you'd need to allocate to hold these buffers.
  virtual int GetMaximumMemorySize() override;

  // How many buffers have been recorded.
  virtual int GetBufferCount() override;

  // Where a given buffer should be placed in the memory arena.
  virtual bool GetOffsetForBuffer(ErrorReporter* error_reporter, int buffer_index, int* offset) override;

  // Prints an ascii-art diagram of the buffer layout plan.
  void PrintMemoryPlan(ErrorReporter* error_reporter);

  // Used to store a list of buffers ordered by their offset.
  struct ListEntry {
    int offset;
    int requirements_index;
    int next_entry_index;
  };

 private:
  // Whether a buffer is active in a given time range.
  bool DoesEntryOverlapInTime(const ListEntry* entry, const int first_time_used, const int last_time_used) const;

  // Walks the list to return the next buffer that is active in a given time
  // range, or a null pointer if there are none.
  ListEntry* NextValidEntry(const ListEntry* start,  const int first_time_used, const int last_time_used);

  // If there isn't an up to date plan, calculate a new one.
  void CalculateOffsetsIfNeeded();

  // How many buffers we can handle. With dynamic memory allocation this can be
  // variable, but for simplicity and the ability to run in an embedded
  // environment, use a hard-coded maximum for now.
  static constexpr int kMaxBufferCount = 1024;

  // Records the client-provided information about each buffer.
  struct BufferRequirements {
    int size;
    int first_time_used;
    int last_time_used;
  };
  BufferRequirements requirements_[kMaxBufferCount];

  // The number of buffers added so far.
  int buffer_count_;

  // Working arrays used during the layout algorithm.
  int buffer_sizes_sorted_by_size_[kMaxBufferCount];
  int buffer_ids_sorted_by_size_[kMaxBufferCount];
  ListEntry buffers_sorted_by_offset_[kMaxBufferCount];
  int next_free_entry_;

  // Stores the outcome of the plan, the location of each buffer in the arena.
  int buffer_offsets_[kMaxBufferCount];

  // Whether buffers have been added since the last plan was calculated.
  bool need_to_calculate_offsets_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MEMORY_PLANNER_GREEDY_MEMORY_PLANNER_H_
