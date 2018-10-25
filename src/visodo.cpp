/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "vo_features.h"
#include <string>
//#include <regex>
#include <iomanip>

using namespace cv;
using namespace std;

#define MAX_FRAME 1709
#define MIN_NUM_FEAT 2000

// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

vector<string> split(string str, string token){
    vector<string>result;
    while(str.size()){
        int index = str.find(token);
        if(index!=string::npos){
            result.push_back(str.substr(0,index));
            str = str.substr(index+token.size());
            if(str.size()==0)result.push_back(str);
        }else{
            result.push_back(str);
            str = "";
        }
    }
    return result;
}

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{

  string line;
  int i = 0;
  ifstream myfile ( "/home/a/Programming/datasets/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv"/*"/home/avisingh/Datasets/KITTI_VO/00.txt"*/);
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  std::string a;
  std::string::size_type sz;

  if (myfile.is_open())
  {
    getline(myfile,line);
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;

      std::istringstream in(line);
      //cout << line << '\n';
      size_t j=0;
      while(j < 4)
      {
        //in >> z;
        in >> a;
        std::vector<string> result = split(a, ",");
        //std::regex wsc_re("[\\s,]+"); // whitespace or comma
        //std::vector<std::string> result{
        //    std::sregex_token_iterator(a.begin(), a.end(), wsc_re, -1), {}
        //};
        //std::cout << std::endl;

        for (size_t iter = 0; iter < 4; ++iter)
        {
            //std::cout << result[iter] << std::endl;
            z = std::stod(result[iter], &sz);
            if (j==2) y=z;
            if (j==1)  x=z;
            ++j;
        }
      }

      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}


int main( int argc, char** argv )	{

  Mat img_1, img_2;
  Mat R_f, t_f; //the final rotation and tranlation vectors containing the
  std::vector<double> qtrnn(4); //0123 = xyzw

  ofstream myfile;
  myfile.open ("pose_out.csv");

  double scale = 1.00;
  char filename1[200];
  char filename2[200];
  sprintf(filename1, "/home/a/Programming/datasets/V1_02_medium/mav0/cam0/data/%19ld.png"/*"/home/avisingh/Datasets/KITTI_VO/00/image_2/%06d.png"*/, 1403715523912143104);
  sprintf(filename2, "/home/a/Programming/datasets/V1_02_medium/mav0/cam0/data/%19ld.png"/*"/home/avisingh/Datasets/KITTI_VO/00/image_2/%06d.png"*/, 1403715523962142976);

  std::ifstream input_ts_file;
  input_ts_file.open("/home/a/Programming/datasets/V1_02_medium/mav0/cam0/timestamps.txt");
  if(!input_ts_file)
  {
      std::cout<<"Error opening ts file"<< std::endl;
      while(1);
  }
  std::string ts_line;
  std::vector<long> eval_timestamps;
  while(std::getline(input_ts_file, ts_line))
  {
      eval_timestamps.push_back(std::stol(ts_line));
  }

  char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;
  cv::Point textOrg(10, 50);

  //read the first two frames from the dataset
  Mat img_1_c = imread(filename1);
  Mat img_2_c = imread(filename2);

  if ( !img_1_c.data || !img_2_c.data ) {
    std::cout<< " --(!) Error reading images " << std::endl; return -1;
  }

  // we work with grayscale images
  cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

  // feature detection, tracking
  vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
  featureDetection(img_1, points1);        //detect features in img_1
  vector<uchar> status;
  featureTracking(img_1,img_2,points1,points2, status); //track those features to img_2

  //TODO: add a fucntion to load these values directly from KITTI's calib files
  // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
  double focal = 457.975;
  cv::Point2d pp(367.215, 248.375);
  //recovering the pose and the essential matrix
  Mat E, R, t, mask;
  E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
  recoverPose(E, points2, points1, R, t, focal, pp, mask);

  Mat prevImage = img_2;
  Mat currImage;
  vector<Point2f> prevFeatures = points2;
  vector<Point2f> currFeatures;

  char filename[100];

  R_f = R.clone();
  t_f = t.clone();
  //std::cout << R.rows << " " << R.cols << std::endl;

  clock_t begin = clock();

  namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

  Mat traj = Mat::zeros(600, 600, CV_8UC3);

  for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)	{
  	sprintf(filename,
            "/home/a/Programming/datasets/V1_02_medium/mav0/cam0/data/%19ld.png"/*"/home/avisingh/Datasets/KITTI_VO/00/image_2/%06d.png"*/,
            eval_timestamps[numFrame]);
    //cout << eval_timestamps[numFrame] << endl;
  	Mat currImage_c = imread(filename);
  	cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	vector<uchar> status;
  	featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

  	E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
  	recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

    Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);


   for(int i=0;i<prevFeatures.size();i++)	{   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
  		prevPts.at<double>(0,i) = prevFeatures.at(i).x;
  		prevPts.at<double>(1,i) = prevFeatures.at(i).y;

  		currPts.at<double>(0,i) = currFeatures.at(i).x;
  		currPts.at<double>(1,i) = currFeatures.at(i).y;
    }

  	scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

    //cout << "Scale is " << scale << endl;

    //if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) { //bogus

      t_f = t_f + 10*scale*(R_f*t);
      R_f = R*R_f;

    //}

    //else {
     //cout << "scale below 0.1, or incorrect translation" << endl;
    //}

    qtrnn[3] = std::sqrt(1.0 + R_f.at<double>(0,0) + R_f.at<double>(1,1) + R_f.at<double>(2,2)) / 2.0;
	double w4 = (4.0 * qtrnn[3]);
	qtrnn[0] = (R_f.at<double>(2,1) - R_f.at<double>(1,2)) / w4 ;
	qtrnn[1] = (R_f.at<double>(0,2) - R_f.at<double>(2,0)) / w4 ;
	qtrnn[2] = (R_f.at<double>(1,0) - R_f.at<double>(0,1)) / w4 ;

    // lines for printing results
    myfile << std::setprecision(19) << eval_timestamps[numFrame] << "," << t_f.at<double>(0) << "," << t_f.at<double>(1) << "," << t_f.at<double>(2) << "," << qtrnn[3] << "," << qtrnn[0] << "," << qtrnn[1] << "," << qtrnn[2] << endl;

    // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
 	if (prevFeatures.size() < MIN_NUM_FEAT)	{
    //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
    //cout << "trigerring redection" << endl;
 	 featureDetection(prevImage, prevFeatures);
    featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);

 	}

    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    int x = int(t_f.at<double>(0)) + 300;
    int y = int(t_f.at<double>(2)) + 100;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

    rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    imshow( "Road facing camera", currImage_c );
    imshow( "Trajectory", traj );

    waitKey(1);

  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;

  //cout << R_f << endl;
  //cout << t_f << endl;

  return 0;
}
