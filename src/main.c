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

static void do_index(const char* dirname)
{
  /* foreach jpg in dirname, compute channel average */

  char filename[256];
  DIR* dirp;
  struct dirent* dent;
  unsigned char rgb[3];
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
    if (strcmp(dent->d_name, ".") == 0) goto skip_index;
    if (strcmp(dent->d_name, "..") == 0) goto skip_index;

    sprintf(filename, "%s/%s", dirname, dent->d_name);

    average_rgb(filename, rgb);

    line_len = sprintf
    (
     line_buf, "%s %02x %02x %02x\n",
     dent->d_name, rgb[0], rgb[1], rgb[2]
    );

    write(index_fd, line_buf, line_len);

  skip_index:
    dent = readdir(dirp);
  }

  close(index_fd);

  closedir(dirp);
}


/* tiling */

struct index_entry
{
  char filename[32];
  unsigned char rgb[3];
  struct index_entry* next;
};

struct index_info
{
  struct index_entry* ie;
  char dirname[32];
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
  char filename[32];
  const char* line;
  struct index_entry* ie;
  struct index_entry* prev_ie = NULL;
  int fd;
  unsigned int rgb[3];

  strcpy(ii->dirname, dirname);
  ii->ie = NULL;

  sprintf(filename, "%s/tilit_index", dirname);

  fd = open(filename, O_RDONLY);

  while ((line = read_line(fd)) != NULL)
  {
    ie = malloc(sizeof(struct index_entry));
    ie->next = NULL;

    sscanf
    (
     line, "%s %02x %02x %02x",
     ie->filename, &rgb[0], &rgb[1], &rgb[2]
    );

    ie->rgb[0] = (unsigned char)rgb[0];
    ie->rgb[1] = (unsigned char)rgb[1];
    ie->rgb[2] = (unsigned char)rgb[2];

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
    free(tmp);
  }
}

static unsigned int compute_dist
(const unsigned char* first_rgb, const unsigned char* second_rgb)
{
  unsigned int d = 0;
  unsigned int i;
  for (i = 0; i < 3; ++i)
  {
    const int diff = first_rgb[i] - second_rgb[i];
    d += diff * diff;
  }
  return d;
}

static struct index_entry* index_find
(struct index_info* ii, const unsigned char* rgb)
{
  struct index_entry* ie = ii->ie;
  unsigned int best_dist;
  struct index_entry* best_ie;

  best_dist = compute_dist(rgb, ie->rgb);
  best_ie = ie;
  ie = ie->next;

  while (ie)
  {
    const unsigned int this_dist = compute_dist(rgb, ie->rgb);

    if (this_dist < best_dist)
    {
      best_dist = this_dist;
      best_ie = ie;
    }

    ie = ie->next;
  }

  return best_ie;
}

static IplImage* do_tile(const char* im_filename, const char* index_dirname)
{
  /* load index_filename */
  /* scale im_filename to 240 */
  /* foreach im pixel, find the most appropriate in index and replace */

  int s;
  IplImage* im_ini;
  IplImage* im_bin;
  IplImage* im_tile;
  IplImage* im_shap;
  CvSize tile_size;
  CvRect tile_roi;
  CvSize shap_size;
  int x;
  int y;
  struct index_info ii;
  struct index_entry* ie;
  unsigned char rgb[3];

  index_load(&ii, index_dirname);

  im_ini = do_open(im_filename);

  /* 64 tiles */
  const int ntil = 64;
  s = ((im_ini->width < ntil) ? ntil : im_ini->width) / ntil;
  im_bin = do_bin(im_ini, s);

  /* 128 pixels per tile */
  const int npix = 128;
  tile_size.width = im_bin->width * npix;
  tile_size.height = im_bin->height * npix;
  im_tile = cvCreateImage(tile_size, IPL_DEPTH_8U, 3);

  shap_size.width = npix;
  shap_size.height = npix;
  im_shap = cvCreateImage(shap_size, IPL_DEPTH_8U, 3);

  for (y = 0; y < im_bin->height; ++y)
    for (x = 0; x < im_bin->width; ++x)
    {
      IplImage* im_near;
      char near_filename[64];

      /* get bined pixel */
      get_pixel(im_bin, x, y, rgb);

      /* find nearest indexed image */
      ie = index_find(&ii, rgb);

      /* reshape nearest image */
      sprintf(near_filename, "%s/%s", ii.dirname, ie->filename);
      im_near = do_open(near_filename);

      printf("found %d %d\n", x, y); fflush(stdout);

      do_reshape(im_near, im_shap);
      cvReleaseImage(&im_near);

      /* blit in tile image */
      tile_roi.x = x * npix;
      tile_roi.y = y * npix;
      tile_roi.width = npix;
      tile_roi.height = npix;

      cvSetImageROI(im_tile, tile_roi);
      cvCopy(im_shap, im_tile, NULL);
      cvResetImageROI(im_tile);
    }

  index_free(&ii);

  cvReleaseImage(&im_bin);
  cvReleaseImage(&im_ini);
  cvReleaseImage(&im_shap);

  return im_tile;
}


int main(int ac, char** av)
{
  if (strcmp(av[1], "index") == 0)
  {
    do_index("../pic/kiosked_2");
  }
  else if (strcmp(av[1], "tile") == 0)
  {
    IplImage* tile_im;
    tile_im = do_tile("../pic/face/main.jpg", "../pic/kiosked");
    cvSaveImage("/tmp/tile.jpg", tile_im, NULL);
    cvReleaseImage(&tile_im);
  }

  return 0;
}
