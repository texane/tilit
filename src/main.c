#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/types.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>


/* tiles count */
#define CONFIG_NTIL 56
/* pixel per tile, makes 10.8mm wide at 300dpi */
#define CONFIG_NPIX 128


static IplImage* do_open(const char* filename)
{
  IplImage* im;
  im = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
  return im;
}


static inline void set_pixel
(IplImage* im, int x, int y, const unsigned char rgb[3])
{
  unsigned char* const p = (unsigned char*)
    (im->imageData + y * im->widthStep + x * 3);
  p[0] = rgb[2];
  p[1] = rgb[1];
  p[2] = rgb[0];
}

static inline void get_pixel
(IplImage* im, int x, int y, unsigned char rgb[3])
{
  const unsigned char* const p = (unsigned char*)
    (im->imageData + y * im->widthStep + x * 3);
  rgb[0] = p[2];
  rgb[1] = p[1];
  rgb[2] = p[0];
}

static inline void get_pixel_rgb
(IplImage* im, int x, int y, unsigned char rgb[3])
{
  get_pixel(im, x, y, rgb);
}

static inline void get_pixel_ycc
(IplImage* im, int x, int y, unsigned char ycc[3])
{
  const unsigned char* const p = (unsigned char*)
    (im->imageData + y * im->widthStep + x * 3);
  ycc[0] = p[0];
  ycc[1] = p[1];
  ycc[2] = p[2];
}

static IplImage* bgr_to_ycc(IplImage* im)
{
  CvSize ycc_size;
  IplImage* ycc_im;

  ycc_size.width = im->width;
  ycc_size.height = im->height;
  ycc_im = cvCreateImage(ycc_size, IPL_DEPTH_8U, 3);
  cvCvtColor(im, ycc_im, CV_BGR2YCrCb);

  return ycc_im;
}

static IplImage* do_bin(IplImage* im, int s)
{
  /* image pixel binning */
  /* s the scaling factor */

  const int ss = s * s;

  IplImage* bin_im = NULL;
  CvSize bin_size;
  int x;
  int y;
  int i;
  int j;
  unsigned int sum[3];
  unsigned char rgb[3];

  bin_size.width = im->width / s - ((im->width % s) ? 1 : 0);
  bin_size.height = im->height / s - ((im->height % s) ? 1 : 0);
  bin_im = cvCreateImage(bin_size, IPL_DEPTH_8U, 3);

  for (y = 0; y < bin_size.height; ++y)
  {
    for (x = 0; x < bin_size.width; ++x)
    {
      sum[0] = 0;
      sum[1] = 0;
      sum[2] = 0;

      for (i = 0; i < s; ++i)
	for (j = 0; j < s; ++j)
	{
	  get_pixel(im, x * s, y * s, rgb);

	  sum[0] += rgb[0];
	  sum[1] += rgb[1];
	  sum[2] += rgb[2];
	}

      rgb[0] = sum[0] / ss;
      rgb[1] = sum[1] / ss;
      rgb[2] = sum[2] / ss;

      set_pixel(bin_im, x, y, rgb);
    }
  }

  return bin_im;
}


static IplImage* do_scale(IplImage* im, int s)
{
  /* image pixel binning */
  /* s the scaling factor */

  IplImage* scale_im = NULL;
  CvSize scale_size;
  int x;
  int y;
  int i;
  int j;
  unsigned char rgb[3];

  scale_size.width = im->width * s;
  scale_size.height = im->height * s;
  scale_im = cvCreateImage(scale_size, IPL_DEPTH_8U, 3);

  for (y = 0; y < im->height; ++y)
  {
    for (x = 0; x < im->width; ++x)
    {
      get_pixel(im, x, y, rgb);

      for (i = 0; i < s; ++i)
	for (j = 0; j < s; ++j)
	  set_pixel(scale_im, x * s + i, y * s + j, rgb);
    }
  }

  return scale_im;
}


static void do_reshape(IplImage* im, IplImage* im_shap)
{
  int ws;
  int hs;
  int ss;
  int x;
  int y;
  int i;
  int j;
  unsigned int sum[3];
  unsigned char rgb[3];

  if (im->width < im_shap->width) return ;
  if (im->height < im_shap->height) return ;

  ws = im->width / im_shap->width;
  hs = im->height / im_shap->height;
  ss = ws * hs;

  for (y = 0; y < im_shap->height; ++y)
    for (x = 0; x < im_shap->width; ++x)
    {
      sum[0] = 0;
      sum[1] = 0;
      sum[2] = 0;

      for (i = 0; i < ws; ++i)
	for (j = 0; j < hs; ++j)
	{
	  get_pixel(im, x * ws, y * hs, rgb);

	  sum[0] += rgb[0];
	  sum[1] += rgb[1];
	  sum[2] += rgb[2];
	}

      rgb[0] = sum[0] / ss;
      rgb[1] = sum[1] / ss;
      rgb[2] = sum[2] / ss;

      set_pixel(im_shap, x, y, rgb);
    }
}


static void do_show(IplImage* im)
{
  static const char* const wname = "fu";
  cvNamedWindow(wname, CV_WINDOW_AUTOSIZE);
  cvShowImage(wname, im);
  cvWaitKey(0);
}


/* build an image directory index */

static void average_rgb(const char* filename, unsigned char* rgb)
{
  IplImage* im;
  int x;
  int y;
  uint64_t sum[3];

  im = do_open(filename);

  sum[0] = 0;
  sum[1] = 0;
  sum[2] = 0;

  for (y = 0; y < im->height; ++y)
    for (x = 0; x < im->width; ++x)
    {
      get_pixel(im, x, y, rgb);
      sum[0] += rgb[0];
      sum[1] += rgb[1];
      sum[2] += rgb[2];
    }

  rgb[0] = sum[0] / (im->width * im->height);
  rgb[1] = sum[1] / (im->width * im->height);
  rgb[2] = sum[2] / (im->width * im->height);

  cvReleaseImage(&im);
}

static void average_ycc(const char* filename, unsigned char* ycc)
{
  IplImage* im;
  IplImage* ycc_im;
  int x;
  int y;
  uint64_t sum[3];

  im = do_open(filename);
  ycc_im = bgr_to_ycc(im);
  cvReleaseImage(&im);

  sum[0] = 0;
  sum[1] = 0;
  sum[2] = 0;

  for (y = 0; y < ycc_im->height; ++y)
    for (x = 0; x < ycc_im->width; ++x)
    {
      get_pixel(ycc_im, x, y, ycc);

      /* actually ccy */
      sum[0] += ycc[2];
      sum[1] += ycc[1];
      sum[2] += ycc[0];
    }

  ycc[0] = sum[0] / (ycc_im->width * ycc_im->height);
  ycc[1] = sum[1] / (ycc_im->width * ycc_im->height);
  ycc[2] = sum[2] / (ycc_im->width * ycc_im->height);

  cvReleaseImage(&ycc_im);
}

static void do_index(const char* dirname)
{
  /* foreach jpg in dirname, compute channel average */

  char filename[256];
  DIR* dirp;
  struct dirent* dent;
  unsigned char rgb[3];
  unsigned char ycc[3];
  int line_len;
  int index_fd;
  char line_buf[256];

  dirp = opendir(dirname);
  if (dirp == NULL) return ;

  sprintf(filename, "%s/%s", dirname, "tilit_index");
  index_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);

  dent = readdir(dirp);
  while (dent != NULL)
  {
    if (strcmp(dent->d_name, "tilit_index") == 0) goto skip_index;
    if (strcmp(dent->d_name, "wget.sh") == 0) goto skip_index;
    if (strcmp(dent->d_name, "wget.py") == 0) goto skip_index;
    if (strcmp(dent->d_name, ".") == 0) goto skip_index;
    if (strcmp(dent->d_name, "..") == 0) goto skip_index;

    sprintf(filename, "%s/%s", dirname, dent->d_name);

    average_rgb(filename, rgb);
    average_ycc(filename, ycc);

    line_len = sprintf
    (
     line_buf, "%s %02x %02x %02x %02x %02x %02x\n",
     dent->d_name,
     rgb[0], rgb[1], rgb[2],
     ycc[0], ycc[1], ycc[2]
    );

    write(index_fd, line_buf, line_len);

  skip_index:
    dent = readdir(dirp);
  }

  close(index_fd);

  closedir(dirp);
}


/* index */

struct index_entry
{
  char filename[128];
  unsigned char rgb[3];
  unsigned char ycc[3];
  unsigned int penalty;
  IplImage* cached_im;
  struct index_entry* next;
};

struct index_info
{
  struct index_entry* ie;
  char dirname[128];
};

static const char* read_line(int fd)
{
  static char line_buf[128];
  unsigned int i;
  for (i = 0; i < sizeof(line_buf) - 1; ++i)
  {
    if (read(fd, line_buf + i, 1) != 1) return NULL;
    if (line_buf[i] == '\n') break ;
  }
  line_buf[i] = 0;
  return line_buf;
}

static void index_load(struct index_info* ii, const char* dirname)
{
  char filename[128];
  const char* line;
  struct index_entry* ie;
  struct index_entry* prev_ie = NULL;
  int fd;
  unsigned int rgb[3];
  unsigned int ycc[3];

  strcpy(ii->dirname, dirname);
  ii->ie = NULL;

  sprintf(filename, "%s/tilit_index", dirname);

  fd = open(filename, O_RDONLY);

  while ((line = read_line(fd)) != NULL)
  {
    ie = malloc(sizeof(struct index_entry));
    ie->next = NULL;
    ie->penalty = 0;
    ie->cached_im = NULL;

    sscanf
    (
     line, "%s %02x %02x %02x %02x %02x %02x",
     ie->filename,
     &rgb[0], &rgb[1], &rgb[2],
     &ycc[0], &ycc[1], &ycc[2]
    );

    ie->rgb[0] = (unsigned char)rgb[0];
    ie->rgb[1] = (unsigned char)rgb[1];
    ie->rgb[2] = (unsigned char)rgb[2];

    ie->ycc[0] = (unsigned char)ycc[0];
    ie->ycc[1] = (unsigned char)ycc[1];
    ie->ycc[2] = (unsigned char)ycc[2];

    if (prev_ie != NULL) prev_ie->next = ie;
    else ii->ie = ie;
    prev_ie = ie;
  }

  close(fd);
}

static void index_free(struct index_info* ii)
{
  struct index_entry* ie = ii->ie;

  while (ie)
  {
    struct index_entry* const tmp = ie;
    ie = ie->next;
    if (tmp->cached_im != NULL) cvReleaseImage(&tmp->cached_im);
    free(tmp);
  }
}

static unsigned int compute_dist
(const unsigned char* a, const unsigned char* b)
{
  unsigned int d = 0;
  unsigned int i;

  static const unsigned int w[] = { 4, 8, 16 };

  for (i = 0; i < 3; ++i)
  {
    const int diff = (a[i] - b[i]) / w[i];
    d += diff * diff;
  }
  return d;
}

static struct index_entry* index_find
(
 struct index_info* ii,
 const unsigned char* rgb,
 const unsigned char* ycc
)
{
  struct index_entry* ie = ii->ie;
  unsigned int best_dist;
  struct index_entry* best_ie;

  best_dist = compute_dist(ycc, ie->ycc);
  best_ie = ie;
  ie = ie->next;

  while (ie)
  {
    if (ie->penalty && (--ie->penalty)) goto skip_ie;

    const unsigned int this_dist = compute_dist(ycc, ie->ycc);

    if (this_dist < best_dist)
    {
      best_dist = this_dist;
      best_ie = ie;
    }

  skip_ie:
    ie = ie->next;
  }

  /* tile can appear 1.5 lines later */
  best_ie->penalty = (3 * CONFIG_NTIL) / 2;

  return best_ie;
}


/* tiler */

struct mozaic_info
{
  struct index_entry* tile_arr;
  int h;
  int w;
  IplImage* tile_im;
  IplImage* ycc_im;
};

static void do_tile
(
 const char* im_filename,
 struct index_info* ii,
 struct mozaic_info* mi
)
{
  int s;
  IplImage* im_ini;
  IplImage* im_bin;
  int x;
  int y;
  unsigned char rgb[3];
  unsigned char ycc[3];

  im_ini = do_open(im_filename);

  /* tile count */
  const int ntil = CONFIG_NTIL;
  const int largest =
    im_ini->width > im_ini->height ? im_ini->width : im_ini->height;
  s = largest / ntil;
  im_bin = do_bin(im_ini, s);

  /* turn into ycc */
  mi->ycc_im = bgr_to_ycc(im_bin);

  /* prepare resulting array */
  mi->w = mi->ycc_im->width;
  mi->h = mi->ycc_im->height;
  mi->tile_arr = malloc(mi->w * mi->h * sizeof(struct index_entry*));

  /* pixels per tile */
  const int npix = CONFIG_NPIX;

  printf("[ do_tile ]\n");

  for (y = 0; y < mi->h; ++y)
  {
    printf("y == %d\n", y); fflush(stdout);

    for (x = 0; x < mi->w; ++x)
    {
      get_pixel_rgb(im_bin, x, y, rgb);
      get_pixel_ycc(mi->ycc_im, x, y, ycc);

      /* find nearest indexed image */
      mi->tile_arr[y * mi->w + x] = index_find(ii, rgb, ycc);
    }
  }

  cvReleaseImage(&im_bin);
  cvReleaseImage(&im_ini);
}


/* image editor */

struct tile_node
{
  int x;
  int y;
  struct tile_node* next;
};

struct ed_info
{
  struct mozaic_info* mi;
  struct index_info* ii;
  IplImage* ed_im;
  int hs;
  int ws;
};

static void redraw_ed(struct ed_info* ei)
{
  /* scale down mi->tile_im to ei->ed_im */

  const int ss = ei->ws * ei->hs;
  int x;
  int y;
  int i;
  int j;
  unsigned int sum[3];
  unsigned char rgb[3];
  struct tile_node* tn;

  for (y = 0; y < ei->ed_im->height; ++y)
  {
    for (x = 0; x < ei->ed_im->width; ++x)
    {
      sum[0] = 0;
      sum[1] = 0;
      sum[2] = 0;

      for (i = 0; i < ei->ws; ++i)
	for (j = 0; j < ei->hs; ++j)
	{
	  get_pixel(im, x * ei->ws, y * ei->hs, rgb);

	  sum[0] += rgb[0];
	  sum[1] += rgb[1];
	  sum[2] += rgb[2];
	}

      rgb[0] = sum[0] / ss;
      rgb[1] = sum[1] / ss;
      rgb[2] = sum[2] / ss;

      set_pixel(ed_im, x, y, rgb);
    }
  }

  /* put rectangles over selected tiles */
  for (tn = ei->sel_tiles; tn; tn = tn->next)
  {
    const CvScalar purple = cvScalar(0xff, 0, 0xff, 0);
    CvPoint points[2];
    const int scaled_x = ;
    const int scaled_y = ;
    points[0] = cvPoint(scaled_x, scaled_y);
    points[1] = cvPoint(scaled_x + , scaled_y + );
    cvRectangle(ui->image, points[0], points[1], purple, 3, 8, 0);
  }
}

static void on_mouse(int event, int x, int y, int flags, void* param)
{
  struct ed_info* const ei = param;

  switch (event)
  {
  case CV_EVENT_LBUTTONDOWN:
    {
      /* tile selection */

      const int tile_x = (x * ei->ws) / CONFIG_NPIX;
      const int tile_y = (y * ei->hs) / CONFIG_NPIX;
      struct tile_node* tn;
      struct tile_node* pre = NULL;

      for (tn = ei->sel_tiles; tn; tn = tn->next)
      {
	if ((tn->x == tile_x) && (tn->y == tile_y))
	  break ;
	pre = tn;
      }

      /* previously unselected */
      if (tn == NULL)
      {
	tn = malloc(sizeof(tile_node));
	tn->x = tile_x;
	tn->y = tile_y;
	tn->next = ei->sel_tiles;
	ei->sel_tiles = tn;
      }
      else /* unselect */
      {
	if (pre) pre->next = tn->next;
	else ei->sel_tiles = tn->next;
	free(tn);
      }

      redraw_ed(ei);

      break ;
    }

  default: break ;
  }
}

static void do_make(struct index_info* ii, struct mozaic_info* mi);

static void do_edit(struct index_info* ii, struct mozaic_info* mi)
{
  struct ed_info ei;
  int is_done = 0;
  int is_update;

  ei.ii = ii;
  ei.mi = mi;
  ei.ed_im = ;
  ei.sel_tiles = NULL;
  ei.ws = ;
  ei.hs = ;

  cvNamedWindow("ed", CV_WINDOW_AUTOSIZE);
  cvShowImage("ed", ei.scaled_im);
  cvSetMouseCallback("ed", on_mouse, (void*)&ei);

  /* foreach tile_arr, add to ei->hist_arr[i] */

  while (is_done == 0)
  {
    const int k = cvWaitKey(0);

    is_update = 0;

    switch (k & 0xff)
    {
    case left:
      {
	/* foreach selected tile */
	/* . if not already, add current to ei->hist_arr[i] list */
	/* . mi->tile_arr[i] = index_find_exclude(ei->hist_arr[i]); */
	/* render in ei->ed_im */
	is_update = 1;
	break ;
      }

    case right:
      {
	/* foreach selected tile */
	/* . if ei->hist_arr[i]->next */
	/* .. mi->tile_arr[i] = ei->hist_arr[i]->cur->next */
	/* .. mi->hist_arr[i]->cur = next */
	is_update = 1;
	break ;
      }

    case 27:
    case enter:
      {
	is_done = 1;
	break ;
      }

    default:
      {
	printf("unknown keycode: %x\n", k);
	break ;
      }
    }

    if (is_update)
    {
      do_make(ei->ii, ei->mi);
      redraw_ed(ei);
    }
  }
}


static void do_make(struct index_info* ii, struct mozaic_info* mi)
{
  CvSize tile_size;
  CvRect tile_roi;
  CvSize shap_size;
  int x;
  int y;

  /* pixels per tile */
  const int npix = CONFIG_NPIX;

  tile_size.width = mi->w * npix;
  tile_size.height = mi->h * npix;
  mi->tile_im = cvCreateImage(tile_size, IPL_DEPTH_8U, 3);

  shap_size.width = npix;
  shap_size.height = npix;

  printf("[ do_make ]\n");

  for (y = 0; y < mi->h; ++y)
  {
    printf("y == %d / %d\n", y, h); fflush(stdout);

    for (x = 0; x < mi->w; ++x)
    {
      struct index_entry* const ie = mi->tile_arr[y * mi->w + x];

      if (ie->cached_im == NULL)
      {
	char near_filename[128];
	IplImage* im_near;

	/* reshape nearest image */
	sprintf(near_filename, "%s/%s", ii->dirname, ie->filename);
	im_near = do_open(near_filename);
	ie->cached_im = cvCreateImage(shap_size, IPL_DEPTH_8U, 3);
	do_reshape(im_near, ie->cached_im);
	cvReleaseImage(&im_near);
      }

      /* blit in tile image */
      tile_roi.x = x * npix;
      tile_roi.y = y * npix;
      tile_roi.width = npix;
      tile_roi.height = npix;

      cvSetImageROI(mi->tile_im, tile_roi);
      cvCopy(ie->cached_im, mi->tile_im, NULL);
    }
  }

  cvResetImageROI(mi->tile_im);
}


int main(int ac, char** av)
{
  if (strcmp(av[1], "index") == 0)
  {
    do_index("../pic/india/trekearth.new/trekearth");
  }
  else if (strcmp(av[1], "tile") == 0)
  {
    struct mozaic_info mi;
    struct index_info ii;

    index_load(&ii, "../pic/india/trekearth.new/trekearth");

    do_tile("../pic/roland_13/main.jpg", &ii, &mi);
    do_make(&ii, &mi);
    do_edit(&ii, &mi);

    cvSaveImage("/tmp/tile.jpg", mi.tile_im, NULL);

    cvReleaseImage(&mi.tile_im);
    cvReleaseImage(&mi.ycc_im);

    free(mi.tile_arr);
    index_free(&ii);
  }

  return 0;
}
