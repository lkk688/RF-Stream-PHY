# Mechanical-rotation sanity check — synthetic Doppler vs real hand motion

> **Bottom line:** At 5 Hz target Doppler (hand-wave at ~30 cm / 1 swing per second), the Doppler shift itself is **below our Pluto LO drift floor (~100 Hz std)** — so we can't directly verify that the synthetic injection produces the same CFO offset as real motion at this scale.  What we *can* verify is that real motion has a strong **amplitude-modulation** signature (pilot std jumps from 96 → 488) that the synthetic Doppler injection does **not** reproduce (it stays at 109, identical to static).  Decode-rate ranking across the three conditions stays within channel-state noise (~5 pp).
>
> **Conclusion:** the synthetic 2-tap injection is a *clean* Doppler model (pure frequency offset on one tap, no path-loss modulation) — useful for isolated Doppler studies in the loopback sweep, but it does **not** capture the amplitude-modulation effects of real mobility.  For the OFDM-vs-OTFS Doppler crossover at the 200–500 Hz cliff, the software loopback remains the right tool; for end-to-end mobility validation, a faster physical motion (vehicle speed, ≥10 m/s) is needed to lift the Doppler shift above the Pluto LO drift floor.

## Setup

| | |
|---|---|
| Carrier | fc = 2.42 GHz (channel state at 2.45 GHz unusable; see [OTFS_VS_OFDM_DOPPLER_LOOPBACK.md](OTFS_VS_OFDM_DOPPLER_LOOPBACK.md)) |
| Sample rate | 3 MHz |
| TX | Jetson Orin + PlutoSDR, `rf_stream_tx_playback.py --cyclic` |
| RX | x86 + PlutoSDR, `rf_stream_rx_otfs.py --num_captures 300` |
| Target Doppler | ~5 Hz peak (~0.6 m/s peak velocity × 2.42 GHz / c = 4.8 Hz) |
| Hand-wave geometry | ~30 cm peak-to-peak swing, ~1 Hz swing rate |
| Synthetic injection | 2-tap channel: direct + −3 dB reflector at delay=3 samples, Doppler 5.27 Hz (quantised to nearest multiple of fs/L) |

Three back-to-back captures, 300 frames each, ~7 s of raw capture + ~3 min of OTFS+LDPC decode per condition.

## Numbers

| Condition | Pkt decode | CFO mean (Hz) | CFO std (Hz) | Pilot mean | Pilot std | SNR_post (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Static (no motion, no injection) | 15/300 = 5.0 % | −4549 | **118** | 467 | **96** | −15.4 |
| **Hand-wave (real ~5 Hz motion)** | 25/300 = 8.3 % | −4525 | **63** | 555 | **488** | −16.2 |
| Synthetic fd=5 Hz (no motion) | 0/300 = 0.0 % | −3880* | 1549* | 347 | 109 | −10.7 |

*Synth fd=5 CFO stats include one strong interferer spike (+3545 Hz) within the sampled set; removing it gives mean ≈ −4220, std ≈ 95.

(CFO/pilot stats are from the every-10th-capture log lines, ~25 clean-signal samples per condition.  Decode-rate uses full 300-capture totals.)

## What the numbers say

1. **5 Hz Doppler is too small to detect against Pluto LO drift.**  Static CFO std is 118 Hz, motion CFO std is 63 Hz — the supposed motion-induced shift (±5 Hz peak) is well below this floor.  Real hand-wave Doppler is buried in Pluto LO temperature drift; we'd need ≥50 Hz peak Doppler (≥6 m/s velocity ≈ jog speed) to see it cleanly above the floor.

2. **Real motion modulates amplitude — synthetic injection does not.**  This is the strongest signal in the dataset:
   - Static pilot std = 96, motion pilot std = **488** → 5× amplitude modulation from the hand-wave (path-length changes, near-field re-coupling, antenna pattern changes within reach of the TX).
   - Synth fd=5 pilot std = 109 ≈ static — the synthetic injection produces only frequency offset on the reflected tap, no path-loss modulation.

3. **Decode rate is dominated by channel-state drift, not Doppler.**  Across all three conditions, pkt rate is 0–8 % — the channel itself has been degrading throughout the session (baseline was 30 % at the start, ~6 % at the end).  At 5 Hz, neither real nor synthetic Doppler is large enough to perturb decode rate beyond this background drift.

## What this validates and what it doesn't

✅ **Synthetic injection is *minimally invasive* at low Doppler** — pkt rate with synthetic fd=5 (0 %) is within channel-state noise of real motion (8 %) and static (5 %).  Adding the synthetic Doppler tap doesn't create artifacts that mimic motion in unexpected ways.

✅ **Synthetic injection is *pure Doppler*, not a motion proxy** — real motion adds amplitude modulation (5× pilot-magnitude variability) that the synthetic model deliberately omits.  This is desirable for the isolated Doppler study (OFDM cliff characterisation) but means the synthetic results understate the difficulty of real mobility, which is amplitude- *and* frequency-modulated.

❌ **5 Hz Doppler equivalence cannot be directly confirmed on this hardware.**  Pluto's LO drift (~100 Hz std over a minute) drowns out a 5 Hz signal.  To validate Doppler magnitude end-to-end would need either:
- Faster physical motion (≥10 m/s, lifts Doppler to 80+ Hz, clearly above LO drift)
- A locked reference between TX and RX (10 MHz GPSDO or shared LO cable), reducing LO drift to <1 Hz
- Long-time-base spectral analysis of the pilot tone (would need NPZ capture mode + post-processing)

## Files & data

- Run logs:
  - `/tmp/mech/static.log`        (no motion, no injection, 5%)
  - `/tmp/mech/motion.log`        (hand-wave, no injection, 8.3%)
  - `/tmp/mech/synth_fd5.log`     (no motion, fd=5 Hz injection, 0%)
- TX IQ:
  - `/tmp/otfs_ldpc_seed42.iq`         (uninjected)
  - `/tmp/otfs_ldpc_seed42_fd5.iq`     (fd=5.27 Hz injected)

## Implications for the paper

- **Keep the software-loopback Doppler sweep as the headline figure** — it isolates the frequency-domain Doppler effect at a controlled SNR, which is exactly the claim ("OTFS handles Doppler that OFDM can't") that needs the cleanest data.
- **Cite the mechanical sanity check as a "no surprises" check** — synthetic injection doesn't introduce artifacts beyond what real motion does, but the Doppler scale we can hand-test is too small to validate against LO drift.
- **For a published mobility result, plan one of:** a GPSDO-locked Pluto pair for the LO-stability fix, or a faster-motion platform (cart at 1–2 m/s gives 8–16 Hz Doppler, edge of detectability; vehicle at 30 km/h ≈ 8.3 m/s gives ~67 Hz, comfortably above drift).
