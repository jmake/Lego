#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <iterator>
#include <iostream>


#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/console/time.h>   // TicToc

#include <Eigen/Dense>


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false;


Eigen::Vector3d 
extractEulerAngles(const Eigen::Matrix4d& transformation) 
{
    // Extract the rotation part (top-left 3x3 matrix)
    Eigen::Matrix3d rotation = transformation.block<3, 3>(0, 0);

    // Convert rotation matrix to Euler angles (ZYX order: yaw, pitch, roll)
    Eigen::Vector3d euler_angles = rotation.eulerAngles(2, 1, 0) * (180.0 / M_PI); // ZYX order
    //Eigen::Vector3d euler_angles_deg = euler_angles * (180.0 / M_PI);

    std::cout << "Euler angles (ZYX order):\n";
    std::cout << "Yaw (Z): " << euler_angles[0] << " deg\n";
    std::cout << "Pitch (Y): " << euler_angles[1] << " deg\n";
    std::cout << "Roll (X): " << euler_angles[2] << " deg\n";
    return euler_angles;
}

void
print4x4Matrix (const Eigen::Matrix4d & matrix)
{
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));

  std::vector<double> values(matrix.data(), matrix.data() + matrix.size());
  std::cout << "Vector values: [";
  for(int i=0; i < 4; i++)
  {
    std::cout << "[ ";
    for(int j=0; j < 3; j++) std::cout<< values[4*i+j] <<" "; 
    std::cout << "] \n";
  }
  //std::copy(values.begin(), values.end(), std::ostream_iterator<double>(std::cout, " "));
  std::cout <<"] " <<std::endl;
}

PointCloudT::Ptr 
pcl_transformPointCloud(
  PointCloudT::Ptr cloud_in, 
  Eigen::Matrix4d transformation_matrix
) 
{
  double theta = M_PI / 8;  // 22.5 deg
  transformation_matrix (0, 0) = std::cos (theta);
  transformation_matrix (0, 1) = -sin (theta);
  transformation_matrix (1, 0) = sin (theta);
  transformation_matrix (1, 1) = std::cos (theta);
  transformation_matrix (2, 3) = 0.4;

  std::cout << "Applying this rigid transformation to: cloud_in -> cloud_icp" << std::endl;
  print4x4Matrix (transformation_matrix);

  PointCloudT::Ptr cloud_icp (new PointCloudT); 
  pcl::transformPointCloud (*cloud_in, *cloud_icp, transformation_matrix);
  return cloud_icp; 
}


Eigen::Matrix4d 
iterative_closest_point_algorithm(
  PointCloudT::Ptr cloud_in, 
  PointCloudT::Ptr &cloud_icp, 
  int iterations 
) 
{
  pcl::console::TicToc time;
  time.tic ();

  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setMaximumIterations (iterations);
  icp.setInputSource (cloud_icp);
  icp.setInputTarget (cloud_in);
  icp.align (*cloud_icp);
  icp.setMaximumIterations (1);  // We set this variable to 1 for the next time we will call .align () function
  std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc () << " ms" << std::endl;

  Eigen::Matrix4d  matrix; 
  if (icp.hasConverged ())
  {
    std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
    std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
    
    Eigen::Matrix4d  matrix = icp.getFinalTransformation ().cast<double>();
    print4x4Matrix (matrix);
    return matrix; 
  }
  else
  {
    PCL_ERROR ("\nICP has not converged.\n");
    return matrix;
  }  
}


PointCloudT::Ptr 
pcl_loadPLYFile(std::string fname) 
{
  pcl::console::TicToc time;
  time.tic ();

  PointCloudT::Ptr cloud (new PointCloudT); 

  if (pcl::io::loadPLYFile (fname, *cloud) < 0)
  {
    PCL_ERROR ("Error loading cloud %s.\n",fname);
    return cloud;
  }
  std::cout << "\nLoaded file " << fname << " (" << cloud->size () << " points) in " << time.toc () << " ms\n" << std::endl;
  return cloud; 
}


int main (int argc, char* argv[])
{
  Eigen::Matrix4d transformation_matrix; 
  transformation_matrix = Eigen::Matrix4d::Identity ();

  PointCloudT::Ptr cloud_in (new PointCloudT);  
  PointCloudT::Ptr cloud_icp (new PointCloudT);  

  cloud_in = pcl_loadPLYFile("untitled1.ply"); 
  cloud_icp = pcl_loadPLYFile("untitled2.ply"); 

  //cloud_icp = pcl_transformPointCloud(cloud_in, transformation_matrix);   
  //pcl::io::savePLYFile("cloud_tr.ply", *cloud_icp);

  transformation_matrix = iterative_closest_point_algorithm(cloud_in, cloud_icp, 100); 
  pcl::io::savePLYFile("cloud_icp.ply", *cloud_icp);

  Eigen::Matrix4d matrix = transformation_matrix.transpose();  
  double determinant = matrix.determinant();
  std::cout << "Determinant: " << determinant << std::endl;  
  std::cout << "Matrix:" << std::endl << matrix << std::endl;
  assert(std::abs(determinant - 1.0) < 1e-4 && "Determinant is not close to 1");

  std::vector<double> values(matrix.data(), matrix.data() + matrix.size());
  assert(std::abs(values[0] - 0.923883) < 1e-4 && "values[0] - 0.923883");
  assert(std::abs(values[11] + 0.4) < 1e-4 && "values[11] + 0.4");

  Eigen::Vector3d euler_angles = extractEulerAngles(transformation_matrix);  
  assert(std::abs(euler_angles[0]  - 157.5) < 1e-4 && "euler_angles[0] -  157.5");

  std::cout << "Vector values: [";
  std::copy(values.begin(), values.end(), std::ostream_iterator<double>(std::cout, " "));
  std::cout <<"] " <<std::endl;
  return (0);
}

/*
rm -rf CMakeCache.txt CMakeFiles Makefile cmake_install.cmake

apt-get install -y cmake cmake-curses-gui build-essential 

*/