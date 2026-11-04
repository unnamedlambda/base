/* zimg and libvips behind three plain functions, so the harness calls them
 * without mirroring their structs. */
#include <stdlib.h>
#include <string.h>
#include <vips/vips.h>
#include <vips/vector.h>
#include <zimg.h>

typedef struct {
	zimg_filter_graph *graph;
	void *tmp;
} zplane;

/* A Lanczos3 graph from one 8-bit plane to another, and its scratch; with
 * `simd` 0 zimg runs its plain C. */
void *zimg_lanczos_new(unsigned sw, unsigned sh, unsigned dw, unsigned dh, int simd)
{
	zimg_image_format s, d;
	zimg_graph_builder_params p;
	zplane *z;
	size_t n;

	zimg_image_format_default(&s, ZIMG_API_VERSION);
	s.width = sw;
	s.height = sh;
	s.pixel_type = ZIMG_PIXEL_BYTE;
	s.depth = 8;
	s.color_family = ZIMG_COLOR_GREY;
	d = s;
	d.width = dw;
	d.height = dh;
	zimg_graph_builder_params_default(&p, ZIMG_API_VERSION);
	p.resample_filter = ZIMG_RESIZE_LANCZOS;
	p.filter_param_a = 3;
	p.cpu_type = simd ? ZIMG_CPU_AUTO : ZIMG_CPU_NONE;
	z = malloc(sizeof *z);
	z->graph = zimg_filter_graph_build(&s, &d, &p);
	if (!z->graph || zimg_filter_graph_get_tmp_size(z->graph, &n))
		return NULL;
	z->tmp = aligned_alloc(64, (n + 63) / 64 * 64);
	return z;
}

/* One plane through the graph; 0 when zimg answered so. */
int zimg_lanczos_run(void *h, const unsigned char *src, long ss, unsigned char *dst, long ds)
{
	zplane *z = h;
	zimg_image_buffer_const s;
	zimg_image_buffer d;

	memset(&s, 0, sizeof s);
	memset(&d, 0, sizeof d);
	s.version = ZIMG_API_VERSION;
	s.plane[0].data = src;
	s.plane[0].stride = ss;
	s.plane[0].mask = ZIMG_BUFFER_MAX;
	d.version = ZIMG_API_VERSION;
	d.plane[0].data = dst;
	d.plane[0].stride = ds;
	d.plane[0].mask = ZIMG_BUFFER_MAX;
	return zimg_filter_graph_process(z->graph, &s, &d, z->tmp, NULL, NULL, NULL, NULL);
}

/* libvips, one worker per pipeline and no operation cache. */
int vips_bench_init(void)
{
	if (VIPS_INIT("image-bench"))
		return -1;
	vips_concurrency_set(1);
	vips_cache_set_max(0);
	return 0;
}

/* Whether libvips may use its SIMD paths. */
void vips_bench_simd(int on)
{
	vips_vector_set_enabled(on);
}

/* One plane halved by vips_reduce with the Lanczos3 kernel, the result
 * copied to `dst`, as vips_image_write_to_memory hands it out. */
int vips_lanczos_plane(const unsigned char *src, int w, int h, unsigned char *dst)
{
	VipsImage *in, *out;
	size_t n;
	void *buf;

	in = vips_image_new_from_memory(src, (size_t)w * h, w, h, 1, VIPS_FORMAT_UCHAR);
	if (!in)
		return -1;
	if (vips_reduce(in, &out, 2.0, 2.0, "kernel", VIPS_KERNEL_LANCZOS3, NULL)) {
		g_object_unref(in);
		return -2;
	}
	buf = vips_image_write_to_memory(out, &n);
	g_object_unref(out);
	g_object_unref(in);
	if (!buf)
		return -3;
	if (n != (size_t)(w / 2) * (h / 2)) {
		g_free(buf);
		return -4;
	}
	memcpy(dst, buf, n);
	g_free(buf);
	return 0;
}
