using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;

namespace CS2RLBot;

public sealed class PolicyInference : IDisposable
{
    // -------------------------------------------------------------------------
    // LatencyTracker — nested, zero-alloc ring buffer for last 64 µs samples
    // -------------------------------------------------------------------------
    public struct LatencyTracker
    {
        private readonly long[] _buf;
        private int _pos;
        private bool _full;

        public LatencyTracker()
        {
            _buf = new long[64];
            _pos = 0;
            _full = false;
        }

        public void Record(long us)
        {
            _buf[_pos] = us;
            _pos = (_pos + 1) % 64;
            if (_pos == 0) _full = true;
        }

        public readonly LatencyStats GetStats()
        {
            int count = _full ? 64 : _pos;
            if (count == 0) return new LatencyStats(0, 0, 0);
            var sorted = _buf[..count].ToArray();
            Array.Sort(sorted);
            long p50  = sorted[(int)(count * 0.50)];
            long p99  = sorted[Math.Max(0, (int)(count * 0.99) - 1)];
            long max  = sorted[count - 1];
            return new LatencyStats(p50, p99, max);
        }

        // Verify the tracker with known inputs. Returns null on success, error string on failure.
        public static string? SelfTest()
        {
            var t = new LatencyTracker();
            // Record 64 values: 1..64 µs
            for (long i = 1; i <= 64; i++) t.Record(i);
            var s = t.GetStats();
            if (s.MaxUs != 64)   return $"SelfTest: expected MaxUs=64, got {s.MaxUs}";
            if (s.P50Us < 32 || s.P50Us > 33) return $"SelfTest: expected P50Us≈32, got {s.P50Us}";
            // Ring wraps: record 65..128, oldest 1..64 should be evicted
            for (long i = 65; i <= 128; i++) t.Record(i);
            var s2 = t.GetStats();
            if (s2.MaxUs != 128) return $"SelfTest(wrap): expected MaxUs=128, got {s2.MaxUs}";
            if (s2.P50Us < 96 || s2.P50Us > 97) return $"SelfTest(wrap): expected P50Us≈96, got {s2.P50Us}";
            return null;
        }
    }

    public void Dispose() { }
}

public record struct LatencyStats(long P50Us, long P99Us, long MaxUs);
