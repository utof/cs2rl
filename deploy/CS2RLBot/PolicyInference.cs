using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

namespace CS2RLBot;

public sealed class PolicyInference : IDisposable
{
    // ── LatencyTracker ────────────────────────────────────────────────────────
    public struct LatencyTracker
    {
        private readonly long[] _buf;
        private int _pos;
        private bool _full;

        public LatencyTracker() { _buf = new long[64]; _pos = 0; _full = false; }

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
            Span<long> sorted = stackalloc long[64];
            _buf.AsSpan(0, count).CopyTo(sorted);
            sorted = sorted[..count];
            MemoryExtensions.Sort(sorted);
            long p50 = sorted[(int)(count * 0.50)];
            long p99 = sorted[Math.Max(0, (int)(count * 0.99) - 1)];
            long max = sorted[count - 1];
            return new LatencyStats(p50, p99, max);
        }

        public static string? SelfTest()
        {
            var t = new LatencyTracker();
            for (long i = 1; i <= 64; i++) t.Record(i);
            var s = t.GetStats();
            if (s.MaxUs != 64) return $"SelfTest: MaxUs expected 64, got {s.MaxUs}";
            if (s.P50Us < 32 || s.P50Us > 33) return $"SelfTest: P50Us expected ~32, got {s.P50Us}";
            for (long i = 65; i <= 128; i++) t.Record(i);
            var s2 = t.GetStats();
            if (s2.MaxUs != 128) return $"SelfTest(wrap): MaxUs expected 128, got {s2.MaxUs}";
            if (s2.P50Us < 96 || s2.P50Us > 97) return $"SelfTest(wrap): P50Us expected ~96, got {s2.P50Us}";
            return null;
        }
    }

    // ── Fields ────────────────────────────────────────────────────────────────
    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions;
    private readonly ILogger _log;

    // Dims — read from session metadata in ctor, never hardcoded
    public readonly int ObsDim;
    public readonly int HiddenDim;
    public readonly int NumHeads;

    // Batch 3: True when the exported ONNX has a continuous-aim head (mu_aim).
    // Driven by `aim_dim > 0` in the sidecar JSON. When false, we fall back to
    // the Batch-2 layout (NumHeads + 2 outputs: logits + LSTM h_out + c_out).
    public readonly bool HasAimHead;

    // Pre-allocated input buffers (reused every inference call — zero alloc)
    private readonly float[] _obs;
    private readonly float[] _done   = new float[1];
    private readonly float[] _lstmH;
    private readonly float[] _lstmC;
    private readonly float[] _lstmHOut;
    private readonly float[] _lstmCOut;

    // Pre-allocated logit output buffers
    private readonly float[][] _actionLogits;

    // Batch 3: continuous-aim head (1D Gaussian Δyaw, μ-only at deploy).
    // Allocated only when HasAimHead = true; empty array sentinel otherwise so
    // callers can safely test `muAim.Length > 0` without null checks.
    // AimDim = 1 matches AIM_DIM in src/_action_spec.py — μ is a scalar Δyaw
    // (in radians, tanh-squashed and scaled to [-π/4, +π/4]).
    private readonly float[]  _muAim;
    public  const    int      AimDim = 1;
    // Note: the OrtValue wrapping _muAim is created in the ctor and stashed in
    // _outputOrts[NumHeads]; disposal happens via the _outputOrts iteration in
    // Dispose(). We deliberately do NOT keep a separate field — that would
    // duplicate the disposal path and invite double-dispose if a future change
    // adds a field-level cleanup.

    // Pinned OrtValue wrappers (created once at ctor, always non-null after successful construction)
    private readonly OrtValue _obsOrt = null!;
    private readonly OrtValue _doneOrt = null!;
    private readonly OrtValue _lstmHOrt = null!;
    private readonly OrtValue _lstmCOrt = null!;
    private readonly OrtValue[] _outputOrts = null!;
    private readonly OrtValue[] _inputValues = null!;
    private readonly string[] _inputNames  = { "obs", "done", "lstm_h", "lstm_c" };
    private readonly string[] _outputNames;

    private LatencyTracker _latencyTracker = new();

    // ── Constructor ───────────────────────────────────────────────────────────
    // hasAimHead: pass true when the ONNX export emits a continuous-aim head
    // (Batch 3+ checkpoints with aim_dim>0 in the sidecar JSON). Pass false to
    // load Batch-2 checkpoints with the legacy NumHeads+2 output layout.
    public PolicyInference(string modelPath, int[] actionSizes, ILogger log, bool hasAimHead = true)
    {
        _log = log;
        HasAimHead = hasAimHead;

        var opts = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads      = 1,
            InterOpNumThreads      = 1,
            ExecutionMode          = ExecutionMode.ORT_SEQUENTIAL,
            EnableMemoryPattern    = true,
            EnableCpuMemArena      = true,
        };
        opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        _session    = new InferenceSession(modelPath, opts);
        _runOptions = new RunOptions();
        opts.Dispose();

        // Read dims from session metadata — no hardcoded values
        var inMeta = _session.InputMetadata;
        ObsDim    = (int)inMeta["obs"].Dimensions[1];       // obs shape: [1, obs_dim]
        HiddenDim = (int)inMeta["lstm_h"].Dimensions[2];   // lstm_h shape: [1, 1, hidden_dim]
        NumHeads  = actionSizes.Length;

        // Output names straight from the session (NOT from JSON sidecar — it doesn't have them)
        _outputNames = _session.OutputMetadata.Keys.ToArray();

        // Batch 3: hybrid policy outputs are logits[N] + mu_aim + lstm_h_out + lstm_c_out
        // = NumHeads + 3. Backward-compat with Batch 2 checkpoints (NumHeads + 2)
        // is allowed when aim_dim=0 in the sidecar (caller passes hasAimHead=false).
        int expectedOutputs = HasAimHead ? NumHeads + 3 : NumHeads + 2;
        if (_outputNames.Length != expectedOutputs)
            throw new InvalidOperationException(
                $"[CS2RLBot] Expected {expectedOutputs} output slots " +
                $"(NumHeads={NumHeads} + {(HasAimHead ? 1 : 0)} aim + 2 LSTM states) " +
                $"but session has {_outputNames.Length}. Actual outputs: [{string.Join(", ", _outputNames)}]");
        for (int _vi = _outputNames.Length - 2; _vi < _outputNames.Length; _vi++)
        {
            var _vn = _outputNames[_vi].ToLowerInvariant();
            if (!_vn.Contains("lstm") && !_vn.Contains("h_out") && !_vn.Contains("c_out"))
                throw new InvalidOperationException(
                    $"[CS2RLBot] Output slot {_vi} ('{_outputNames[_vi]}') does not look like an LSTM " +
                    $"hidden-state output (expected name containing 'lstm', 'h_out', or 'c_out'). " +
                    $"Actual outputs: [{string.Join(", ", _outputNames)}]");
        }

        // Allocate input buffers
        _obs      = new float[ObsDim];
        _lstmH    = new float[HiddenDim];
        _lstmC    = new float[HiddenDim];
        _lstmHOut = new float[HiddenDim];
        _lstmCOut = new float[HiddenDim];

        // Allocate logit output buffers
        _actionLogits = new float[NumHeads][];
        for (int i = 0; i < NumHeads; i++)
            _actionLogits[i] = new float[actionSizes[i]];

        // Batch 3: allocate mu_aim only when the policy actually emits it.
        // Empty-array sentinel keeps RunInference's tuple shape stable for callers.
        _muAim = HasAimHead ? new float[AimDim] : Array.Empty<float>();

        // Pin input/output OrtValues (shapes must match export-time dynamic_axes)
        // Use local nullable variables so that partial construction never calls Dispose on
        // an uninitialized handle; assign to readonly fields only after all succeed (C1)
        OrtValue? obsOrt = null, doneOrt = null, lstmHOrt = null, lstmCOrt = null, muAimOrt = null;
        // Batch 3: outputOrts layout when HasAimHead=true:
        //   [0..NumHeads-1] = logits, [NumHeads] = mu_aim,
        //   [NumHeads+1] = lstm_h_out, [NumHeads+2] = lstm_c_out.
        // When HasAimHead=false, the mu_aim slot is omitted (Batch-2 layout).
        int totalOutputs = HasAimHead ? NumHeads + 3 : NumHeads + 2;
        var outputOrts = new OrtValue[totalOutputs];
        try
        {
            obsOrt   = OrtValue.CreateTensorValueFromMemory(_obs,   new long[] { 1, ObsDim });
            doneOrt  = OrtValue.CreateTensorValueFromMemory(_done,  new long[] { 1 });
            lstmHOrt = OrtValue.CreateTensorValueFromMemory(_lstmH, new long[] { 1, 1, HiddenDim });
            lstmCOrt = OrtValue.CreateTensorValueFromMemory(_lstmC, new long[] { 1, 1, HiddenDim });

            // Pin output OrtValues
            for (int i = 0; i < NumHeads; i++)
                outputOrts[i] = OrtValue.CreateTensorValueFromMemory(
                    _actionLogits[i], new long[] { 1, actionSizes[i] });
            int idx = NumHeads;
            if (HasAimHead)
            {
                muAimOrt = OrtValue.CreateTensorValueFromMemory(_muAim, new long[] { 1, AimDim });
                outputOrts[idx++] = muAimOrt;
            }
            outputOrts[idx++] = OrtValue.CreateTensorValueFromMemory(_lstmHOut, new long[] { 1, 1, HiddenDim });
            outputOrts[idx++] = OrtValue.CreateTensorValueFromMemory(_lstmCOut, new long[] { 1, 1, HiddenDim });
        }
        catch
        {
            obsOrt?.Dispose(); doneOrt?.Dispose(); lstmHOrt?.Dispose(); lstmCOrt?.Dispose();
            muAimOrt?.Dispose();
            foreach (var o in outputOrts) o?.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
            throw;
        }
        // Assign to readonly fields only after all creations succeeded
        _obsOrt   = obsOrt;
        _doneOrt  = doneOrt;
        _lstmHOrt = lstmHOrt;
        _lstmCOrt = lstmCOrt;
        // muAimOrt (when non-null) is already stored in outputOrts[NumHeads];
        // the local goes out of scope but the array reference keeps it alive.
        _outputOrts  = outputOrts;
        _inputValues = new[] { _obsOrt, _doneOrt, _lstmHOrt, _lstmCOrt };

        _log.LogInformation(
            "[CS2RLBot] PolicyInference init — model={Model} obs_dim={ObsDim} hidden_dim={HiddenDim} " +
            "num_heads={NumHeads} outputs=[{Outputs}] ort_version={OrtVer}",
            modelPath, ObsDim, HiddenDim, NumHeads,
            string.Join(",", _outputNames),
            typeof(InferenceSession).Assembly.GetName().Version);
    }

    // ── Inference ─────────────────────────────────────────────────────────────
    /// <summary>
    /// Runs one inference step. Returns logit arrays (one per discrete head) plus
    /// the continuous mu_aim buffer (Batch 3+). When HasAimHead=false, MuAim is an
    /// empty array (callers can branch on `MuAim.Length > 0`).
    /// Caller must NOT hold onto the returned arrays across inference calls — they
    /// are reused on the next RunInference call.
    /// </summary>
    public (float[][] Logits, float[] MuAim) RunInference(ReadOnlySpan<float> obs, bool isDone)
    {
        obs.CopyTo(_obs.AsSpan());
        _done[0] = isDone ? 1f : 0f;

        var sw = Stopwatch.GetTimestamp();
        _session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputOrts);
        long elapsedUs = (Stopwatch.GetTimestamp() - sw) * 1_000_000 / Stopwatch.Frequency;

        _latencyTracker.Record(elapsedUs);

        if (elapsedUs > 1000)
            _log.LogWarning("[CS2RLBot] Inference latency {Us}µs exceeded 1ms threshold", elapsedUs);

        // Propagate LSTM state: output → input for next call
        Buffer.BlockCopy(_lstmHOut, 0, _lstmH, 0, HiddenDim * sizeof(float));
        Buffer.BlockCopy(_lstmCOut, 0, _lstmC, 0, HiddenDim * sizeof(float));

        return (_actionLogits, _muAim);
    }

    // ── LSTM state ────────────────────────────────────────────────────────────
    public void ResetLstmState()
    {
        // Log norm before clearing (useful for verifying state has been accumulating)
        double norm = 0;
        for (int i = 0; i < HiddenDim; i++) norm += _lstmH[i] * _lstmH[i];
        _log.LogDebug("[CS2RLBot] ResetLstmState — \u2016h\u2016 before clear = {Norm:F4}", Math.Sqrt(norm));

        Array.Clear(_lstmH);
        Array.Clear(_lstmC);
    }

    // ── Stats ─────────────────────────────────────────────────────────────────
    public LatencyStats GetStats() => _latencyTracker.GetStats();

    // ── Dispose ───────────────────────────────────────────────────────────────
    public void Dispose()
    {
        // Null-tolerant: fields are null! sentinels if the constructor threw before assignment (C1)
        _obsOrt?.Dispose();
        _doneOrt?.Dispose();
        _lstmHOrt?.Dispose();
        _lstmCOrt?.Dispose();
        if (_outputOrts != null)
            foreach (var o in _outputOrts)
                o?.Dispose();
        _runOptions?.Dispose();
        _session?.Dispose();
    }
}

public record struct LatencyStats(long P50Us, long P99Us, long MaxUs);
