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

#include "greedy_memory_planner.h"

#include <cstdio>

#include "reverse_sort_in_place.h"

namespace tflite {

GreedyMemoryPlanner::GreedyMemoryPlanner() : buffer_count_(0), need_to_calculate_offsets_(true) {}
GreedyMemoryPlanner::~GreedyMemoryPlanner() {}

bool GreedyMemoryPlanner::AddBuffer(tflite::ErrorReporter* error_reporter, int size, int first_time_used, int last_time_used) {
  if (buffer_count_ >= kMaxBufferCount) {
    error_reporter->Report("Too many buffers (max is %d)", kMaxBufferCount);
    return false;
  }
  BufferRequirements* current = &requirements_[buffer_count_];
  current->size = size;
  current->first_time_used = first_time_used;
  current->last_time_used = last_time_used;
  ++buffer_count_;
  need_to_calculate_offsets_ = true;
  return true;
}

bool GreedyMemoryPlanner::DoesEntryOverlapInTime(const GreedyMemoryPlanner::ListEntry* entry, const int first_time_used, const int last_time_used) const {
  const BufferRequirements* entry_requirements = &requirements_[entry->requirements_index];
  if (entry_requirements->first_time_used > last_time_used) {
    return false;
  }
  if (first_time_used > entry_requirements->last_time_used) {
    return false;
  }
  return true;
}

GreedyMemoryPlanner::ListEntry* GreedyMemoryPlanner::NextValidEntry(const GreedyMemoryPlanner::ListEntry* start, const int first_time_used, const int last_time_used) {
  if ((start == nullptr) || (start->next_entry_index == -1)) {
    return nullptr;
  }
  ListEntry* result = nullptr;
  ListEntry* candidate_next_entry = &buffers_sorted_by_offset_[start->next_entry_index];
  do {
    if (DoesEntryOverlapInTime(candidate_next_entry, first_time_used, last_time_used)) {
      result = candidate_next_entry;
      break;
    }
    if (candidate_next_entry->next_entry_index == -1) {
      break;
    }
    candidate_next_entry = &buffers_sorted_by_offset_[candidate_next_entry->next_entry_index];
  } while (true);
  return result;
}

void GreedyMemoryPlanner::CalculateOffsetsIfNeeded() {
  if (!need_to_calculate_offsets_ || (buffer_count_ == 0)) {
    return;
  }
  need_to_calculate_offsets_ = false;

  // Start off by ordering the buffers in descending order of size.
  // This helps find a more compact layout. Intuitively, you can think
  // about putting the large buffers in place first, and then the
  // smaller buffers can fit in the gaps, rather than fragmenting the
  // gaps with small buffers at the beginning.
  for (int i = 0; i < buffer_count_; ++i) {
    buffer_sizes_sorted_by_size_[i] = requirements_[i].size;
    buffer_ids_sorted_by_size_[i] = i;
  }
  // This sorting algorithm is naive, and may end up taking a very long time
  // with hundreds of buffers.
  ReverseSortInPlace(buffer_sizes_sorted_by_size_, buffer_ids_sorted_by_size_, buffer_count_);

  // Put the largest buffer at offset zero to start the process.
  ListEntry* first_entry = &buffers_sorted_by_offset_[0];
  first_entry->offset = 0;
  first_entry->requirements_index = buffer_ids_sorted_by_size_[0];
  first_entry->next_entry_index = -1;
  next_free_entry_ = 1;

  // Work through the rest of the buffers to find a good gap to place each one.
  for (int i = 1; i < buffer_count_; ++i) {
    // The id is the order the buffer was originally added by the client.
    const int buffer_id = buffer_ids_sorted_by_size_[i];
    // Look at what size and time range the buffer needs to be active.
    BufferRequirements* wanted_requirements = &requirements_[buffer_id];
    const int wanted_size = wanted_requirements->size;
    const int wanted_first_time_used = wanted_requirements->first_time_used;
    const int wanted_last_time_used = wanted_requirements->last_time_used;
    // Find the first buffer that's active in our time range. All placed
    // buffers are stored in the order of their starting position in the arena
    // so that it's easy to find the next buffer in memory, and so the gap.
    // The candidate_entry variable holds the buffer that we're considering
    // placing the current buffer after.
    ListEntry* candidate_entry;
    if (DoesEntryOverlapInTime(first_entry, wanted_first_time_used, wanted_last_time_used)) {
      candidate_entry = first_entry;
    } else {
      candidate_entry = NextValidEntry(first_entry, wanted_first_time_used, wanted_last_time_used);
    }
    // Loop through the offset-ordered list of buffers, looking for gaps.
    while (true) {
      // Find out what the next active buffer is.
      ListEntry* next_entry = NextValidEntry(candidate_entry, wanted_first_time_used, wanted_last_time_used);
      if (next_entry == nullptr) {
        // We're at the end of the list, so we can always append the buffer
        // here.
        break;
      }
      BufferRequirements* candidate_requirements = &requirements_[candidate_entry->requirements_index];
      // Find out how much space there is between us and the next buffer.
      const int gap = next_entry->offset - (candidate_entry->offset + candidate_requirements->size);
      if (gap >= wanted_size) {
        // This entry has a big enough gap between it and the next, so
        // use it!
        break;
      }
      // The gap wasn't big enough, so move on to another candidate.
      candidate_entry = next_entry;
    }
    // At this point, we've either found a gap (possibly at the end of the
    // list) and want to place the buffer there, or there are no other active
    // buffers in this time range and so we can put it at offset zero.
    int offset;
    if (candidate_entry != nullptr) {
      BufferRequirements* candidate_requirements = &requirements_[candidate_entry->requirements_index];
      offset = (candidate_entry->offset + candidate_requirements->size);
    } else {
      offset = 0;
    }
    // Record the buffer's offset in our plan.
    buffer_offsets_[buffer_id] = offset;
    // Add the newly-placed buffer to our offset-ordered list, so that
    // subsequent passes can fit in their buffers around it.
    ListEntry* new_entry = &buffers_sorted_by_offset_[next_free_entry_];
    new_entry->offset = offset;
    new_entry->requirements_index = buffer_id;
    const int new_entry_index = next_free_entry_;
    ++next_free_entry_;
    ListEntry* current_entry = first_entry;
    // Make sure that we insert the buffer at the correct place in the ordered
    // list.
    while (true) {
      const int next_entry_index = current_entry->next_entry_index;
      if (next_entry_index == -1) {
        // We're at the end of the list, so just add the new entry here.
        current_entry->next_entry_index = new_entry_index;
        new_entry->next_entry_index = -1;        
        break;
      }
      ListEntry* next_entry = &buffers_sorted_by_offset_[next_entry_index];
      if (next_entry->offset > offset) {
        // We're at the right spot to do an insertion and retain the sorting
        // order, so place the new entry here.
        new_entry->next_entry_index = current_entry->next_entry_index;
        current_entry->next_entry_index = new_entry_index;
        break;
      }
      current_entry = next_entry;
    }
  }
}

int GreedyMemoryPlanner::GetMaximumMemorySize() {
  CalculateOffsetsIfNeeded();
  if (buffer_count_ == 0) {
    return 0;
  }
  ListEntry* entry = &buffers_sorted_by_offset_[0];
  int max_size = 0;
  while (entry) {
    BufferRequirements* requirements = &requirements_[entry->requirements_index];
    const int current_size = entry->offset + requirements->size;
    if (current_size > max_size) {
      max_size = current_size;
    }
    if (entry->next_entry_index == -1) {
      break;
    }
    entry = &buffers_sorted_by_offset_[entry->next_entry_index];
  }
  return max_size;
}

void GreedyMemoryPlanner::PrintMemoryPlan(ErrorReporter* error_reporter) {
  CalculateOffsetsIfNeeded();
  constexpr int kLineWidth = 80;
  int max_size = kLineWidth;
  int max_time = 0;
  for (int i = 0; i < buffer_count_; ++i) {
    BufferRequirements* requirements = &requirements_[i];
    const int offset = buffer_offsets_[i];
    const int last_time_used = requirements->last_time_used;
    const int size = offset + requirements->size;
    if (size > max_size) {
      max_size = size;
    }
    if (last_time_used > max_time) {
      max_time = last_time_used;
    }
  }

  char line[kLineWidth + 1];
  for (int t = 0; t <= max_time; ++t) {
    for (int c = 0; c < kLineWidth; ++c) {
      line[c] = '.';
    }
    for (int i = 0; i < buffer_count_; ++i) {
      BufferRequirements* requirements = &requirements_[i];
      if ((t < requirements->first_time_used) || (t > requirements->last_time_used)) {
        continue;
      }
      const int offset = buffer_offsets_[i];
      const int size = requirements->size;
      const int line_start = (offset * kLineWidth) / max_size;
      const int line_end = ((offset + size) * kLineWidth) / max_size;
      for (int n = line_start; n < line_end; ++n) {
        if (line[n] == '.') {
          line[n] = '0' + (i % 10);
        } else {
          line[n] = '!';
        }
      }
    }
    line[kLineWidth] = 0;
    error_reporter->Report("%s", line);
  }
}


int GreedyMemoryPlanner::GetBufferCount() { return buffer_count_; }

bool GreedyMemoryPlanner::GetOffsetForBuffer(tflite::ErrorReporter* error_reporter, int buffer_index, int* offset) {
  if ((buffer_index < 0) || (buffer_index >= buffer_count_)) {
      error_reporter->Report("buffer index %d is outside range 0 to %d", buffer_index, buffer_count_);
      return false;
  }
  *offset = buffer_offsets_[buffer_index];
  return true;
}

}  // namespace tflite
