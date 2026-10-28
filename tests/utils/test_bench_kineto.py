import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


_TESTING_PATH = Path(__file__).parents[2] / 'deep_ep' / 'utils' / 'testing.py'
_TESTING_SPEC = importlib.util.spec_from_file_location('deep_ep_testing', _TESTING_PATH)
testing = importlib.util.module_from_spec(_TESTING_SPEC)
_TESTING_SPEC.loader.exec_module(testing)


class _CpuEventTable:

    def __init__(self, events):
        self.events = events

    def table(self, **kwargs):
        kwargs['sort_by'] = 'cpu_time_total'
        return self.events.table(**kwargs)


class _CpuProfilerAdapter:

    def __init__(self, profiler):
        self.profiler = profiler
        self.steps = 0

    def __enter__(self):
        self.profiler.__enter__()
        return self

    def __exit__(self, *args):
        return self.profiler.__exit__(*args)

    def step(self):
        self.steps += 1
        self.profiler.step()

    def key_averages(self):
        return _CpuEventTable(self.profiler.key_averages())


class BenchKinetoTest(unittest.TestCase):

    def test_kineto_uses_one_warmup_and_num_tests_recorded_callbacks(self):
        real_profile = torch.profiler.profile
        real_activity = torch.profiler.ProfilerActivity
        real_ones = torch.ones
        profilers = []
        calls = []
        flushes = []
        barriers = []
        synchronizations = []

        def cpu_profile(activities, **kwargs):
            self.assertEqual(activities, [real_activity.CPU])
            adapter = _CpuProfilerAdapter(real_profile(activities=activities, **kwargs))
            profilers.append(adapter)
            return adapter

        def cpu_ones(*args, **kwargs):
            return real_ones(*args, **{**kwargs, 'device': 'cpu'})

        def fn():
            calls.append(None)
            with torch.profiler.record_function('kineto_test_kernel'):
                pass

        with patch.object(torch.profiler, 'ProfilerActivity', SimpleNamespace(CUDA=real_activity.CPU)), \
             patch.object(torch.profiler, 'profile', side_effect=cpu_profile), \
             patch.object(torch, 'ones', side_effect=cpu_ones), \
             patch.object(torch.cuda, 'synchronize', side_effect=lambda: synchronizations.append(None)), \
             patch.object(torch.cuda, '_sleep', lambda _: None, create=True), \
             patch.object(testing, 'flush_l2_cache', side_effect=lambda enabled=True: flushes.append(enabled)), \
             patch.dict(os.environ, {'EP_USE_NVIDIA_TOOLS': '0', 'EP_DISABLE_BARRIER_PROFILING': '0'}):
            for num_tests in (30, 1):
                with self.subTest(num_tests=num_tests):
                    calls.clear()
                    flushes.clear()
                    barriers.clear()
                    synchronizations.clear()

                    testing.bench_kineto(
                        fn,
                        'kineto_test_kernel',
                        num_tests=num_tests,
                        flush_l2=True,
                        barrier_comm_profiling=True,
                        barrier=lambda: barriers.append(None),
                    )

                    self.assertEqual(len(calls), num_tests + 2)  # Preflight, warmup, and measured callbacks.
                    self.assertEqual(len(flushes), num_tests + 1)
                    self.assertTrue(all(flushes))
                    self.assertEqual(len(barriers), num_tests + 1)
                    self.assertEqual(len(synchronizations), 3)
                    self.assertEqual(profilers[-1].steps, 2)
                    event_counts = {event.key: event.count for event in profilers[-1].profiler.key_averages()}
                    self.assertEqual(event_counts['kineto_test_kernel'], num_tests)


if __name__ == '__main__':
    unittest.main()
