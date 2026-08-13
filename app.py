
import streamlit as st
import numpy as np
import io, os, struct, time, pandas as pd, warnings, contextlib, sys, tempfile, shutil
from PIL import Image
from bitarray import bitarray
from collections import Counter

# Streamlit magic output is avoided: intermediate arrays/matrices are never
# left as standalone expressions; only final user-facing results are rendered.
import matplotlib.pyplot as plt

# note: JPEG-LS comparison removed from this build

warnings.filterwarnings("ignore")

# ---------------------------
# Helpers
# ---------------------------
@contextlib.contextmanager
def suppress_output():
    """Temporarily silence stdout/stderr (used during heavy loops)."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()

# ---------------------------
# Range coder (24-bit state)
# ---------------------------
FULL, HALF, QUARTER, THREE_QUARTER = 1 << 24, 1 << 23, 1 << 22, 3 << 22

class RangeEncoder:
    def __init__(self):
        self.low, self.high, self.pending, self.out_bits = 0, FULL - 1, 0, []

    def _emit_bit(self, b):
        self.out_bits.append(b)
        while self.pending > 0:
            self.out_bits.append(1 - b)
            self.pending -= 1

    def encode(self, sym, cum, total):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum[sym + 1]) // total - 1
        self.low  = self.low + (rng * cum[sym]) // total
        while True:
            if self.high < HALF:
                self._emit_bit(0); self.low <<= 1; self.high = (self.high << 1) | 1
            elif self.low >= HALF:
                self._emit_bit(1); self.low = (self.low - HALF) << 1; self.high = ((self.high - HALF) << 1) | 1
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low = (self.low - QUARTER) << 1
                self.high = ((self.high - QUARTER) << 1) | 1
            else:
                break

    def finish(self):
        self._emit_bit(0 if self.low < QUARTER else 1)

    def to_bytes(self):
        b = bitarray(self.out_bits)
        while len(b) % 8: b.append(0)
        return b.tobytes()

class RangeDecoder:
    def __init__(self, data):
        self.low, self.high, self.code = 0, FULL - 1, 0
        self.data = bitarray(endian='big'); self.data.frombytes(data)
        self.pos = 0
        for _ in range(24):
            self.code = (self.code << 1) | self._bit()
    def _bit(self):
        if self.pos >= len(self.data): return 0
        b = 1 if self.data[self.pos] else 0
        self.pos += 1
        return b
    def decode(self, cum, total):
        rng = self.high - self.low + 1
        val = ((self.code - self.low + 1) * total - 1) // rng
        sym = 0
        while cum[sym + 1] <= val: sym += 1
        self.high = self.low + (rng * cum[sym + 1]) // total - 1
        self.low  = self.low + (rng * cum[sym]) // total
        while True:
            if self.high < HALF: pass
            elif self.low >= HALF:
                self.code -= HALF; self.low -= HALF; self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.code -= QUARTER; self.low -= QUARTER; self.high -= QUARTER
            else: break
            self.low <<= 1; self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._bit()
        return sym

# ---------------------------
# Predictor, context and color decorrelation (YCoCg)
# ---------------------------
SCALE = 256

def predict_pixel_weighted(x, y, img):
    A = int(img[y, x-1]) if x>0 else 0
    B = int(img[y-1, x]) if y>0 else 0
    C = int(img[y-1, x-1]) if (x>0 and y>0) else 0
    grad_h = abs(A - C); grad_v = abs(B - C)
    wa = SCALE // (1 + grad_v); wb = SCALE // (1 + grad_h)
    return (A*wa + B*wb)//(wa+wb) if (wa+wb) else (A+B)//2

def context_for_pixel(x, y, img):
    A = int(img[y, x-1]) if x>0 else 0
    B = int(img[y-1, x]) if y>0 else 0
    g = abs(A - B)
    return 0 if g < 4 else 1 if g < 16 else 2 if g < 64 else 3

def rgb_to_ycocg_uint16(img):
    R, G, B = img[...,0].astype(np.int32), img[...,1].astype(np.int32), img[...,2].astype(np.int32)
    Co = R - B
    tmp = B + (Co >> 1)
    Cg = G - tmp
    Y = tmp + (Cg >> 1)
    return np.stack([Y, Co + 512, Cg + 512], axis=2).astype(np.uint16)

def ycocg_to_rgb_uint8(ycocg):
    Y = ycocg[...,0].astype(np.int32)
    Co = ycocg[...,1].astype(np.int32) - 512
    Cg = ycocg[...,2].astype(np.int32) - 512
    tmp = Y - (Cg >> 1)
    G = Cg + tmp
    B = tmp - (Co >> 1)
    R = B + Co
    return np.clip(np.stack([R,G,B], axis=2), 0, 255).astype(np.uint8)

# ---------------------------
# Channel compress/decompress (contextual)
def compress_channel_contextual(img_channel, num_contexts=4, offset=2048):
    h, w = img_channel.shape
    ctx_res = [[] for _ in range(num_contexts)]
    for y in range(h):
        for x in range(w):
            pred = predict_pixel_weighted(x,y,img_channel)
            res = int(img_channel[y,x]) - pred
            ctx_res[context_for_pixel(x,y,img_channel)].append(res + offset)
    ctx_symbols, ctx_counts = [], []
    for lst in ctx_res:
        freq = Counter(lst)
        symbols = sorted(freq.keys())
        counts = [freq[s] for s in symbols]
        ctx_symbols.append(symbols); ctx_counts.append(counts)
    ctx_cum, ctx_totals = [], []
    for counts in ctx_counts:
        cum = [0]
        for c in counts:
            cum.append(cum[-1] + c)
        ctx_cum.append(cum); ctx_totals.append(cum[-1])
    enc = RangeEncoder()
    for y in range(h):
        for x in range(w):
            pred = predict_pixel_weighted(x,y,img_channel)
            res = int(img_channel[y,x]) - pred
            s = res + offset
            ctx = context_for_pixel(x,y,img_channel)
            idx = ctx_symbols[ctx].index(s)
            enc.encode(idx, ctx_cum[ctx], ctx_totals[ctx])
    enc.finish()
    return enc.to_bytes(), ctx_symbols, ctx_counts, offset, h, w

def decompress_channel_contextual(data, ctx_symbols, ctx_counts, offset, h, w):
    ctx_cum, ctx_totals = [], []
    for counts in ctx_counts:
        cum = [0]
        for c in counts:
            cum.append(cum[-1] + c)
        ctx_cum.append(cum); ctx_totals.append(cum[-1])
    dec = RangeDecoder(data)
    img = np.zeros((h,w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            ctx = context_for_pixel(x,y,img)
            if ctx_totals[ctx] == 0:
                val = 0
            else:
                idx = dec.decode(ctx_cum[ctx], ctx_totals[ctx])
                val = ctx_symbols[ctx][idx] - offset
            pred = predict_pixel_weighted(x,y,img)
            img[y,x] = int(np.clip(pred + val, 0, 65535))
    return img

# ---------------------------
# Main processing function
# ---------------------------
def process_images(files):
    if not files:
        return {
            "df": pd.DataFrame(),
            "visuals": [],
            "csv_bytes": b"",
            "zip_bytes": b"",
            "per_items": [],
            "charts": []
        }

    temp_dir = tempfile.mkdtemp(prefix="caprcppp_")
    rows, visuals, per_links = [], [], []

    for f in files:
        try:
            file_name = f.name
            file_bytes = f.getvalue()
            base, ext = os.path.splitext(os.path.basename(file_name))
            img = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
            h, w = img.shape[:2]
            orig_kb = len(file_bytes) / 1024.0

            # optional YCoCg decorrelation (we always use it here)
            proc = rgb_to_ycocg_uint16(img)

            with suppress_output():
                t0 = time.time()
                channels = [compress_channel_contextual(proc[...,i].astype(np.int32)) for i in range(3)]
                comp_time = time.time() - t0

            # write binary container
            bin_path = os.path.join(temp_dir, f"{base}_CAPRCppp.bin")
            with open(bin_path, "wb") as fb:
                fb.write(b"CAPX")
                for (d, ctx_symbols, ctx_counts, off, ch_h, ch_w) in channels:
                    fb.write(struct.pack("<IIq", ch_h, ch_w, off))
                    for ctx in range(4):
                        fb.write(struct.pack("<I", len(ctx_symbols[ctx])))
                        for s,c in zip(ctx_symbols[ctx], ctx_counts[ctx]):
                            fb.write(struct.pack("<qq", s, c))
                    fb.write(struct.pack("<I", len(d)))
                    fb.write(d)
            comp_kb = os.path.getsize(bin_path) / 1024.0

            # decompress from container
            t1 = time.time()
            rec_proc = np.zeros((h,w,3), dtype=np.uint16)
            with open(bin_path, "rb") as fb:
                fb.read(4)
                for ch in range(3):
                    ch_h, ch_w, off_r = struct.unpack("<IIq", fb.read(16))
                    ctx_symbols = []; ctx_counts = []
                    for _ in range(4):
                        n_sym = struct.unpack("<I", fb.read(4))[0]
                        syms=[]; cnts=[]
                        for _ in range(n_sym):
                            sv, cv = struct.unpack("<qq", fb.read(16))
                            syms.append(sv); cnts.append(cv)
                        ctx_symbols.append(syms); ctx_counts.append(cnts)
                    dlen = struct.unpack("<I", fb.read(4))[0]
                    data_bytes = fb.read(dlen)
                    rec_ch = decompress_channel_contextual(data_bytes, ctx_symbols, ctx_counts, off_r, ch_h, ch_w)
                    rec_proc[..., ch] = np.clip(rec_ch, 0, 65535).astype(np.uint16)
            dec_time = time.time() - t1

            rec_rgb = ycocg_to_rgb_uint8(rec_proc)
            rec_path = os.path.join(temp_dir, f"{base}_reconstructed.png")
            Image.fromarray(rec_rgb).save(rec_path)

            residual = np.abs(img.astype(np.int16) - rec_rgb.astype(np.int16)).mean(axis=2)
            mse = np.mean((img.astype(np.float32) - rec_rgb.astype(np.float32))**2)
            psnr = float('inf') if mse==0 else 20*np.log10(255.0/np.sqrt(mse))
            cr = orig_kb / comp_kb if comp_kb>0 else 0
            bpp = (comp_kb*1024*8) / (h*w*3) if (h*w)>0 else 0

            # JPEG-LS comparison removed
            jls_cr = jls_psnr = None

            # visuals
            fig, ax = plt.subplots(1,3, figsize=(14,5))
            _im0 = ax[0].imshow(img)
            ax[0].set_title("Original")
            _im1 = ax[1].imshow(rec_rgb)
            ax[1].set_title("Reconstructed")
            _im2 = ax[2].imshow(residual, cmap="inferno")
            ax[2].set_title("Residual Map")
            for a in ax:
                _axis_result = a.axis("off")
            fig.suptitle(f"{base}{ext} | CR={cr:.2f}× | PSNR={psnr:.2f} dB")
            visuals.append({
                "name": f"{base}_comparison.png",
                "bytes": fig_to_png_bytes(fig)
            })
            plt.close(fig)

            rows.append({
                "Image": f"{base}{ext}",
                "Resolution": f"{w}×{h}",
                "Original (KB)": f"{orig_kb:.2f}",
                "Compressed (KB)": f"{comp_kb:.2f}",
                "CR (CAP-RC++)": f"{cr:.2f}",
                "PSNR (CAP-RC++)": f"{psnr:.2f} dB",
                "CompTime": f"{comp_time:.2f}s",
                "DecompTime": f"{dec_time:.2f}s",
               
            })

            per_links.append({
                "Image": f"{base}{ext}",
                "Reconstructed PNG": rec_path,
                "Compressed BIN": bin_path
            })

        except Exception as e:
            st.warning(f"Error processing {getattr(f, 'name', str(f))}: {e}")

    if not rows:
        return {
            "df": pd.DataFrame(),
            "visuals": [],
            "csv_bytes": b"",
            "zip_bytes": b"",
            "per_items": [],
            "charts": []
        }

    # Dataframe for table
    df = pd.DataFrame(rows)
    csv_path = os.path.join(temp_dir, "CAPRCppp_Results.csv")
    df.to_csv(csv_path, index=False)

    # zip archive of temp dir
    zip_path = shutil.make_archive(os.path.join(temp_dir, "CAPRCppp_all"), "zip", temp_dir)

    # Charts (only if multiple images)
    charts = []
    if len(df) > 1:
        # numeric conversions for plotting
        def to_float_col(series):
            # remove non-numeric chars and coerce
            return pd.to_numeric(series.astype(str).str.replace(r"[^\d\.\-eE]", "", regex=True), errors="coerce").fillna(0.0)
        cr_col = to_float_col(df["CR (CAP-RC++)"])
        psnr_col = to_float_col(df["PSNR (CAP-RC++)"])
        # CR chart
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(df["Image"], cr_col)
        ax.set_ylabel("Compression Ratio (×)")
        ax.set_title("Compression Ratio (CAP-RC+++)")
        cp = os.path.join(temp_dir, "CR_chart.png")
        fig.savefig(cp, bbox_inches="tight")
        plt.close(fig)
        charts.append(cp)
        # PSNR chart
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(df["Image"], psnr_col)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("PSNR (CAP-RC++)")
        pp = os.path.join(temp_dir, "PSNR_chart.png")
        fig.savefig(pp, bbox_inches="tight")
        plt.close(fig)
        charts.append(pp)

        # JPEG-LS comparison charts removed

    with open(csv_path, "rb") as f_csv:
        csv_bytes = f_csv.read()
    with open(zip_path, "rb") as f_zip:
        zip_bytes = f_zip.read()

    per_items = []
    for p in per_links:
        rec_path = p["Reconstructed PNG"]
        bin_path = p["Compressed BIN"]
        with open(rec_path, "rb") as f_png:
            rec_png_bytes = f_png.read()
        with open(bin_path, "rb") as f_bin:
            bin_bytes = f_bin.read()
        # Keep only user-facing download/display data in the returned result.
        # The raw compression matrices/symbol lists are never sent to Streamlit.
        original_name = p["Image"]
        original_bytes = None
        for original_file in files:
            if getattr(original_file, "name", "") == original_name:
                original_bytes = original_file.getvalue()
                break

        per_items.append({
            "image": original_name,
            "original_bytes": original_bytes,
            "rec_png_name": os.path.basename(rec_path),
            "rec_png_bytes": rec_png_bytes,
            "bin_name": os.path.basename(bin_path),
            "bin_bytes": bin_bytes
        })

    chart_items = []
    for chart_path in charts:
        with open(chart_path, "rb") as f_chart:
            chart_items.append({
                "name": os.path.basename(chart_path),
                "bytes": f_chart.read()
            })

    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "df": df,
        "visuals": visuals,
        "csv_bytes": csv_bytes,
        "zip_bytes": zip_bytes,
        "per_items": per_items,
        "charts": chart_items
    }


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="CAP-RC+++ | Lossless Image Compression",
    page_icon="🗜️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, application-style CSS. No compression matrices or debug output are
# rendered anywhere in the UI.
st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem 0 1rem 0;
    }

    .hero h1 {
        margin-bottom: 0.25rem;
        font-size: 2.2rem;
    }

    .hero p {
        color: #9aa4b2;
        font-size: 1rem;
        margin-top: 0;
    }

    .metric-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 1rem;
        min-height: 105px;
        background: rgba(128,128,128,.05);
    }

    .metric-label {
        color: #9aa4b2;
        font-size: .85rem;
        margin-bottom: .35rem;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
    }

    .success-box {
        border: 1px solid #35b37e;
        border-radius: 10px;
        padding: .8rem 1rem;
        margin: .75rem 0 1rem 0;
        background: rgba(53,179,126,.08);
    }

    .section-title {
        margin-top: 1.4rem;
        margin-bottom: .8rem;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🗜️ CAP-RC+++</h1>
    <p>Context-Adaptive Lossless Image Compression</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Upload BMP, TIFF, or PNG images. Compress them, verify exact reconstruction, "
    "compare sizes, and download the compressed output."
)

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["bmp", "tif", "tiff", "png"],
    accept_multiple_files=True,
    help="You can upload one or more lossless images."
)

if uploaded_files:
    names = ", ".join(f.name for f in uploaded_files[:5])
    extra = f" + {len(uploaded_files) - 5} more" if len(uploaded_files) > 5 else ""
    st.info(f"Selected {len(uploaded_files)} image(s): {names}{extra}")

if st.button("🚀 Start Compression", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    else:
        with st.spinner("Compressing and reconstructing images..."):
            results = process_images(uploaded_files)

        df = results["df"]

        if df.empty:
            st.error("No images could be processed.")
        else:
            st.success(
                f"Compression completed successfully for {len(df)} image(s)."
            )

            # Overall summary
            st.markdown('<div class="section-title"><h2>Compression Results</h2></div>',
                        unsafe_allow_html=True)

            if len(df) == 1:
                row = df.iloc[0]

                def metric(label, value):
                    return f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """

                # Extract numeric values from the formatted result row.
                original_kb = float(row["Original (KB)"])
                compressed_kb = float(row["Compressed (KB)"])
                ratio = float(row["CR (CAP-RC++)"])
                psnr_text = str(row["PSNR (CAP-RC++)"])
                compression_time = str(row["CompTime"])
                decompression_time = str(row["DecompTime"])

                if original_kb > 0:
                    saved_pct = max(0.0, (1 - compressed_kb / original_kb) * 100)
                else:
                    saved_pct = 0.0

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(metric("Original Size", f"{original_kb:.2f} KB"),
                                unsafe_allow_html=True)
                with c2:
                    st.markdown(metric("Compressed Size", f"{compressed_kb:.2f} KB"),
                                unsafe_allow_html=True)
                with c3:
                    st.markdown(metric("Compression Ratio", f"{ratio:.2f}×"),
                                unsafe_allow_html=True)
                with c4:
                    st.markdown(metric("Space Saved", f"{saved_pct:.2f}%"),
                                unsafe_allow_html=True)

                if psnr_text.startswith("inf"):
                    verification = "PASS — Exact lossless reconstruction"
                else:
                    verification = f"Check reconstruction (PSNR: {psnr_text})"

                st.markdown(
                    f'<div class="success-box">✅ <b>Lossless Verification:</b> {verification}</div>',
                    unsafe_allow_html=True
                )

                # Image comparison
                item = results["per_items"][0]
                st.markdown('<div class="section-title"><h2>Image Comparison</h2></div>',
                            unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        item["original_bytes"],
                        caption=f"Original — {item['image']}",
                        use_container_width=True
                    )
                with col2:
                    st.image(
                        item["rec_png_bytes"],
                        caption="Reconstructed — lossless output",
                        use_container_width=True
                    )

                with st.expander("Processing Details"):
                    d1, d2 = st.columns(2)
                    d1.write(f"**Resolution:** {row['Resolution']}")
                    d1.write(f"**Compression time:** {compression_time}")
                    d2.write(f"**Decompression time:** {decompression_time}")
                    d2.write(f"**PSNR:** {psnr_text}")

                st.markdown('<div class="section-title"><h2>Downloads</h2></div>',
                            unsafe_allow_html=True)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "⬇️ Download Compressed File",
                        data=item["bin_bytes"],
                        file_name=item["bin_name"],
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                with d2:
                    st.download_button(
                        "⬇️ Download Reconstructed PNG",
                        data=item["rec_png_bytes"],
                        file_name=item["rec_png_name"],
                        mime="image/png",
                        use_container_width=True
                    )

            else:
                # Multi-image mode: show a compact summary, not raw arrays.
                st.dataframe(
                    df[[
                        "Image", "Resolution", "Original (KB)",
                        "Compressed (KB)", "CR (CAP-RC++)",
                        "PSNR (CAP-RC++)", "CompTime", "DecompTime"
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown('<div class="section-title"><h2>Image Results</h2></div>',
                            unsafe_allow_html=True)

                for idx, (row, item) in enumerate(
                    zip(df.to_dict("records"), results["per_items"])
                ):
                    with st.expander(f"🖼️ {row['Image']}", expanded=(idx == 0)):
                        original_kb = float(row["Original (KB)"])
                        compressed_kb = float(row["Compressed (KB)"])
                        ratio = float(row["CR (CAP-RC++)"])
                        saved_pct = max(
                            0.0,
                            (1 - compressed_kb / original_kb) * 100
                        ) if original_kb else 0.0

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Original", f"{original_kb:.2f} KB")
                        m2.metric("Compressed", f"{compressed_kb:.2f} KB")
                        m3.metric("Ratio", f"{ratio:.2f}×")
                        m4.metric("Space Saved", f"{saved_pct:.2f}%")

                        if str(row["PSNR (CAP-RC++)"]).startswith("inf"):
                            st.success("✅ Lossless verification passed — exact reconstruction.")

                        img1, img2 = st.columns(2)
                        with img1:
                            st.image(item["original_bytes"], caption="Original",
                                     use_container_width=True)
                        with img2:
                            st.image(item["rec_png_bytes"], caption="Reconstructed",
                                     use_container_width=True)

                        dl1, dl2 = st.columns(2)
                        with dl1:
                            st.download_button(
                                "⬇️ Compressed BIN",
                                data=item["bin_bytes"],
                                file_name=item["bin_name"],
                                mime="application/octet-stream",
                                key=f"bin_{idx}"
                            )
                        with dl2:
                            st.download_button(
                                "⬇️ Reconstructed PNG",
                                data=item["rec_png_bytes"],
                                file_name=item["rec_png_name"],
                                mime="image/png",
                                key=f"png_{idx}"
                            )

                st.download_button(
                    "⬇️ Download CSV Summary",
                    data=results["csv_bytes"],
                    file_name="CAPRCppp_Results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.download_button(
                    "📦 Download ZIP (All Files)",
                    data=results["zip_bytes"],
                    file_name="CAPRCppp_all.zip",
                    mime="application/zip",
                    use_container_width=True
                )

st.markdown("---")
st.caption(
    "CAP-RC+++ • Context-Adaptive Lossless Image Compression • "
    "Intermediate pixel values and compression matrices are intentionally hidden."
)
