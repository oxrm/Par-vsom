// epil-par-vsom.cu/
// version 3.0
// Author(s):####
//
// This file constitues a set of routines which are useful in constructing
// and evaluating self-organizing maps (SOMs)in a GPU environment.


// The application allows the user to define the size of the maps and the number
//iterations as part of the arguments:

//Usage: vsom_executable.exe [X_size]... [Y_size]... [Number _iter]...
//asumming we want a 150 x 100 amps with a 100 iterations we run:

//vsom_executable.exe 150 100 1000


// License
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is available at
//   https://www.gnu.org/licenses

//Load Libraries
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <iostream>
#include <fstream>      // std::ifstream
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <ctime>
#include <cmath>
#include <algorithm>    // std::max
#include <vector>
#include <omp.h>

//For random sample
#include <random>
#include <algorithm>
#include <iterator>

///Prototypes/////////////////////////////////////////////////////////

// We'll use a 2-tuple to store our 2d vector coordinates types
typedef thrust::tuple<float,float> Float2;

// This functor implements the RowSum for 2d vectors
struct RowSum_2d : public thrust::unary_function<Float2,float>
{
    __host__ __device__
        float operator()(const Float2& a) const
        {
            return thrust::get<0>(a) + thrust::get<1>(a);
        }
};


// This functor implements hood func
struct hood_func
{
     int nei_size;
     hood_func(int n_s) : nei_size(n_s) {};
     __host__ __device__
     float operator()(float x) const

     {

      if ( sqrt(x) < nei_size * 1.5)
       {return 1;}
      else
       {return 0;}

     }

};


//This functor implements the update M (updates the neuron weiaghts)
struct update_m_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<3>(t) = thrust::get<3>(t) - (thrust::get<1>(t) * (thrust::get<0>(t))) * thrust::get<2>(t) ;
    }
};



//This functor implements find the winning neuron (BMU)
struct find_bmu_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<2>(t) = pow(thrust::get<0>(t) - thrust::get<1>(t),2) ;
    }
};


// We'll use a 8-tuple to store our 8d vector type
typedef thrust::tuple<float,float,float,float,float,float,float,float> Float8;
// This functor implements the RowSum for 8d vectors
struct RowSum_8d : public thrust::unary_function<Float8,float>
{
    __host__ __device__
        float operator()(const Float8& a) const
        {
            return thrust::get<0>(a) + thrust::get<1>(a) +
                   thrust::get<2>(a) + thrust::get<3>(a) +
				   thrust::get<4>(a) + thrust::get<5>(a) +
				   thrust::get<6>(a) + thrust::get<7>(a) ;
        }
};

//This functor executes the PI and Delta transforming the coordinates
struct find_nei_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<2>(t) = pow(thrust::get<0>(t) - thrust::get<1>(t),2) ;
    }
};


//functor for X <- 1^n x X_k
struct saxpy_functor
{
   // const float a;
   float *a;

    saxpy_functor(float *_a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const  float& y) const {

     return a[0];

        }
};


//Function Definitions///////////////////////////////////////////////////////

// Return a host vector with random values in the range (0,1)
thrust::host_vector<float> random_vector(const size_t N,
  unsigned int seed = thrust::default_random_engine::default_seed)
{
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    thrust::host_vector<float> temp(N);
    for(size_t i = 0; i < N; i++) {
        temp[i] = u01(rng);
    }
    return temp;
}

////Main program sequence of the Par-VSOM

//the executable receives 3 argumentes (x size, y size, number of iters)
int main(int argc, char *argv[])
{

int  x_val;
x_val=atoi(argv[1]);
int  y_val;
y_val=atoi(argv[2]);
int  iters_val;
iters_val=atoi(argv[3]);


int N;
int new_n = x_val * y_val;
N = new_n;

//Init the GPU before start processing
cudaFree(0);



////////////////////////////////////////////////////////////////////////////////
//Init CLocks
std::clock_t c_start = std::clock();
std::clock_t c_end ;

//Define device (GPU) vectors to hold training data

//Define device vector to hold training data (8 columns)
thrust::device_vector < float > file_d_0;
thrust::device_vector < float > file_d_1;
thrust::device_vector < float > file_d_2;
thrust::device_vector < float > file_d_3;
thrust::device_vector < float > file_d_4;
thrust::device_vector < float > file_d_5;
thrust::device_vector < float > file_d_6;
thrust::device_vector < float > file_d_7;

//Define device iterators to create tuple (8 columns matrix)
typedef thrust::device_vector<float>::iterator                  FloatIterator_d;
typedef thrust::
tuple<FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d>  FloatIteratorTuple_d;
typedef thrust::zip_iterator<FloatIteratorTuple_d>              Float8Iterator_d;


srand((unsigned)time(NULL));


//Vectors declaration

//Trainig vector D (l x d)
thrust::device_vector < float > file_vector_d;

//Neuron vector M (n x d)
thrust::device_vector<float> M_vector_d = random_vector(N*8,rand() % 100000 + 1);

//Vector X (Holds the selected training instance)
//  'X' vector N*8=size, and init to 0
thrust::device_vector<float> X_vector_d(N*8,0);
//  'X' vector N*8=size, and init to 0
thrust::host_vector<float> X_vector_h(N*8,0);

//Vector Delta 'D'
// 'D' vector 'Delta' N*8=size, and init to 0
thrust::device_vector<float> D_vector_d(N*8,0);

//Vector PI 'P'
// vector 'P' vector N*8=size, and init to 0
thrust::device_vector<float> P_vector_d(N*8,0);

 // The 'S' vector N=size, and init to 0
thrust::device_vector<float> S(N,0);

//Vector Pc 'P_coordinates' for gamma function
//vector Pc N*2, coordinates for BMU x and y in one vector
thrust::device_vector<float> P_coors(N*2,0);

 //vector is a merger of all coordinate x and y  vector N*2 =size, and init to 0
thrust::device_vector<float> P_coors_all(N*2,0);

//Vector PI in coordinates (gamma)
//vector to contain the PI coordinates
thrust::device_vector<float> PI_coors(N*2,0);

//Vector D_coor_d big vector (holds all the distances)
// n_0 components of the 'D_coor' vector N=size, and init to 0
thrust::device_vector<float> D_coor_d(N,0);

//Vector hood to  nei  (gamma)
//vector to contain the nei (1,0)
thrust::device_vector<float> hood_vec_d(N,0);

//vector to contain the nei (1,0)
thrust::device_vector<float> hood_vec_d_dim(N*8,0);

//Mask vector in GPU
thrust::device_vector<float> mask_vec_d(N,0);

//Nei cache (Huge vector containing the nei for each instance)
thrust::device_vector<float> nei_cache(new_n * N,0);


//Vector for Eta values
thrust::device_vector<float> Eta_d(N*8,0);
// Fill Eta with eta value
thrust::fill(Eta_d.begin(), Eta_d.end(), 0.7);

//Create P Matrix ( holds the coordinates)
 // n_0 (x) components of the 'P_coor' vector N=size, and init to 0
thrust::device_vector<float> P_coor0(N,0);
  // n_1 (y) components of the 'P_coor' vector N=size, and init to 0
thrust::device_vector<float> P_coor1(N,0);

//Init P Matrix wtih sequence for calculations (holds the coordinates)
thrust::sequence(P_coor0.begin(), P_coor0.end());
thrust::sequence(P_coor1.begin(), P_coor1.end());

//Create Operator Matrix for coordinate calculation
// fill with x size values to calculate coordinate system
thrust::device_vector<float> Oper_row(N);
thrust::fill(Oper_row.begin(), Oper_row.end(), x_val);


thrust::device_vector<float> Oper_col(N);
thrust::fill(Oper_col.begin(), Oper_col.end(), x_val);

//Calculate X coordinates
thrust::transform(P_coor0.begin(), P_coor0.end(), Oper_row.begin(), P_coor0.begin(), thrust::divides<int>());

//Calculate y coordinates
thrust::transform(P_coor1.begin(), P_coor1.end(), Oper_col.begin(), P_coor1.begin(), thrust::modulus<int>());

//Init of vector for cooridnate merge in big vector
P_coors_all.resize(P_coors.size());
thrust::copy(P_coor0.begin(),P_coor0.end(),P_coors_all.begin());
thrust::copy(P_coor1.begin(),P_coor1.end(),P_coors_all.begin()+N);

//Create the C Matrix
// n_0 components of the 'C' vector N=size, and init to 0
thrust::device_vector<float> C0_coor(N,0);
  // n_1 components of the 'C' vector N=size, and init to 0
thrust::device_vector<float> C1_coor(N,0);


//Define device iterators to create tuple coordinates
typedef thrust::device_vector<float>::iterator                 FloatIterator_c;
typedef thrust::tuple<FloatIterator_c, FloatIterator_c>   FloatIteratorTuple_c;
typedef thrust::zip_iterator<FloatIteratorTuple_c>            Float2Iterator_c;


//Iterator for Row2 coordinates
Float2Iterator_c first_PI_coor  = thrust::make_zip_iterator(thrust::make_tuple(PI_coors.begin(),PI_coors.begin() + N));
Float2Iterator_c last_PI_coor   = thrust::make_zip_iterator(thrust::make_tuple(PI_coors.begin() + N,PI_coors.end()));

// Create iterator for C Matrix (type Float2Iterator)
Float2Iterator_c first_C = thrust::make_zip_iterator(thrust::make_tuple(C0_coor.begin(),C1_coor.begin()));
Float2Iterator_c last_C  = thrust::make_zip_iterator(thrust::make_tuple(C0_coor.end(),C1_coor.end()));


//Create Delta_coor Matrix //////////////////////////////////////////////////////////////////////////////
  // n_0 components of the 'C' vector N=size, and init to 0
thrust::device_vector<float> D0_coor(N,0);
// n_1 components of the 'C' vector N=size, and init to 0
thrust::device_vector<float> D1_coor(N,0);

// Create iterator for D Matrix (type Float2Iterator)
Float2Iterator_c first_D_coor = thrust::make_zip_iterator(thrust::make_tuple(D0_coor.begin(),D1_coor.begin()));
Float2Iterator_c last_D_coor  = thrust::make_zip_iterator(thrust::make_tuple(D0_coor.end(),D1_coor.end()));

//Create PI_coor Matrix ///////////////////////////////////////////////////////////////////////////////////
 // n_0 components of the 'Pi_coor' vector N=size, and init to 0
thrust::device_vector<float> P0_coor(N,0);
// n_1 components of the 'Pi_coor' vector N=size, and init to 0
thrust::device_vector<float> P1_coor(N,0);

Float2Iterator_c first_P_coor  = thrust::make_zip_iterator(thrust::make_tuple(P0_coor.begin(),P1_coor.begin()));
Float2Iterator_c last_P_coor   = thrust::make_zip_iterator(thrust::make_tuple(P0_coor.end(),P1_coor.end()));

//Create D_coor Matrix ///////////////////////////////////////////////////////////////////////////////////
// n_0 components of the 'D_coor' vector N=size, and init to
thrust::device_vector<float> D_coor(N,0);

//Create hood vector
 // n_0 components of the 'D_coor' vector N=size, and init to 0
thrust::device_vector<float> hood_vec(N,0);


//Iterator of M Matrix (Random Neuron Weights) for debug
Float8Iterator_d first_M1= thrust::make_zip_iterator(thrust::make_tuple(M_vector_d.begin(),M_vector_d.begin() + (N * 1),M_vector_d.begin() + (N * 2),M_vector_d.begin() + (N*3),
                                                    M_vector_d.begin() + (N*4),M_vector_d.begin() + (N*5),M_vector_d.begin() + (N*6),M_vector_d.begin() + (N*7)));
Float8 m_1 = first_M1[0];

//std::cout << "Print out M Matrix (Random Weights)" << std::endl;
//for(size_t i = 0; i < 10; i++)
//  {
//          m_1 = first_M1[i];
//          std::cout << "(" << thrust::get<0>(m_1) << "," << thrust::get<1>(m_1) << "," << thrust::get<2>(m_1) << "," << thrust::get<3>(m_1) << ")" << std::endl;
//    }


//Random variable for Xk (use size of dataset + 1)
int v1 = rand() % 235 + 1;

//Initialize the Matrix size to read Matrix D Data
float file_mat[236][8];

//read data file and load into array
    std::ifstream reader("epil.data");
    if (!reader)
    std::cerr << "Error opening file";
    else
    {
       for (int i = 0; i < 236; i++)
       {  for (int j = 0; j < 8; j++)
         {
              reader >> file_mat[i][j];
              reader.ignore();

         }

       }


     }

     //load data into huge vector
       for ( int j = 0; j < 8; j++)
       {  for (int i = 0; i < 236; i++)
          {
           file_vector_d.push_back(file_mat[i][j]);
          }
        }

//Define neurons file other variables before epocs loop
std::ofstream neuronsFile;
//Init train iterations
int train = iters_val;

//Init index of c as 0
int c_index = 0;
//Declare x,y as int
int x,y;
//Declare tuple
thrust::pair<thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator> tuple_v;

//Variables for neighborhood calculations
int max_val = max(x_val,y_val);
int nei_size = max_val + 1;
float temp_val = (float)train/nei_size;
int nei_step = ceil((float)temp_val);
int nei_counter = 0;

if (nei_step == 0)
{
    nei_step = 1;
}

Float8Iterator_d vector_first_D = thrust::make_zip_iterator(thrust::make_tuple(P_vector_d.begin(),P_vector_d.begin() + (N * 1),P_vector_d.begin() + (N * 2),P_vector_d.begin() + (N*3),
                                  P_vector_d.begin() + (N*4),P_vector_d.begin() + (N*5),P_vector_d.begin() + (N*6),P_vector_d.begin() + (N*7)));

Float8Iterator_d vector_last_D  = thrust::make_zip_iterator(thrust::make_tuple(P_vector_d.begin() + (N * 1), P_vector_d.begin() + (N * 2) ,P_vector_d.begin() + (N *3),P_vector_d.begin() + (N*4),
                                  M_vector_d.begin() + (N*5),M_vector_d.begin() + (N*6),M_vector_d.begin() + (N*7),M_vector_d.begin() + (N*8)));

//Par-VSOM Main training  "Epocs" loops/////////////////////////////////////////
for (int epocs = 0 ; epocs < train; epocs++)
{

//Verify if we need to reduce neighborhood size///////////
nei_counter = nei_counter + 1;
if (nei_counter == nei_step)
{

    nei_counter = 0;
    nei_size = nei_size - 1;

    //Clear the masking cache array
    thrust::fill(mask_vec_d.begin(), mask_vec_d.end() ,0);
}

v1 = rand() % 235 + 1;


//This creates the D vector Repeting the raw_pointer value in the range specified by X_vector (saves repetion value in D_vector)

//D1
thrust::transform(X_vector_d.begin(), X_vector_d.begin()+(N * 1), X_vector_d.begin(),X_vector_d.begin(), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[v1]))));
//D2
thrust::transform(X_vector_d.begin() + ( N * 1), X_vector_d.begin()+(N * 2), X_vector_d.begin() +( N * 1),X_vector_d.begin()+(N * 1), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[236 * 1 + v1]))));
//D3
thrust::transform(X_vector_d.begin()+ ( N * 2), X_vector_d.begin()+(N * 3), X_vector_d.begin() + ( N * 2) ,X_vector_d.begin()+(N * 2), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[236 * 2 + v1]))));
//D4
thrust::transform(X_vector_d.begin() +( N * 3), X_vector_d.begin()+(N * 4), X_vector_d.begin() +(N * 3),X_vector_d.begin()+(N * 3), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[236 * 3 + v1]))));
//D5
thrust::transform(X_vector_d.begin() +( N * 4), X_vector_d.begin()+(N * 5), X_vector_d.begin() +(N * 4),X_vector_d.begin()+(N * 4), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[236 * 4 + v1]))));
//D6
thrust::transform(X_vector_d.begin() +( N * 5), X_vector_d.begin()+(N * 6), X_vector_d.begin() +(N * 5),X_vector_d.begin()+(N * 5), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[236 * 5 + v1]))));
//D7
thrust::transform(X_vector_d.begin() +( N * 6), X_vector_d.begin()+(N * 7), X_vector_d.begin() +(N * 6),X_vector_d.begin()+(N * 6), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[236 * 6 + v1]))));
//D8
thrust::transform(X_vector_d.begin() +( N * 7), X_vector_d.begin()+(N * 8), X_vector_d.begin() +(N * 7),X_vector_d.begin()+(N * 7), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[236 * 7 + v1]))));


//Calculate delta vector :
thrust::transform(M_vector_d.begin(), M_vector_d.end(), X_vector_d.begin(), D_vector_d.begin(), thrust::minus<float>());

//Calculate PI vector:
thrust::transform(D_vector_d.begin(), D_vector_d.end(), D_vector_d.begin(), P_vector_d.begin(), thrust::multiplies<float>());

//Calculate S Vector with Rowsum reduction and store in S vector
thrust::transform(vector_first_D, vector_last_D, S.begin(), RowSum_8d());

//Find minimum
tuple_v = thrust::minmax_element(S.begin(),S.end());
c_index = (tuple_v.first - S.begin());

//Done with BMU//////////////////////////////////////////////////////////////////////////////////

//Check if neighborhood is cached
if (mask_vec_d[c_index] == 1)
{
 thrust::copy(nei_cache.begin() + (new_n * c_index),nei_cache.begin()+ (new_n * c_index) + N,hood_vec_d.begin());
}

//if not in cache Calculate the nei
else
{
 mask_vec_d[c_index] = 1;


//Start of Update neighborhood /////////////////////////////////////////////////
//Gamma Function /////


//Init PC x
x = P_coor0[c_index];

//Init PC y
y = P_coor1[c_index];


//Fill Matrix C with index coordinates X <-- 1^m * Pc
thrust::fill(P_coors.begin(), P_coors.begin()+ N, x);
thrust::fill(P_coors.begin()+ N, P_coors.end(), y);


//Apply the find coordinates neighborhood functor transformation (PI and Delta) to the coodinates vector (vector with X and Y)
thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(P_coors_all.begin(),P_coors.begin(),PI_coors.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(P_coors_all.end(),P_coors.end(),PI_coors.end())),
                     find_nei_functor());

// Finally, we pass the zip_iterators into transform() as if they
// were 'normal' iterators for a device_vector<Float2>.
thrust::transform(first_PI_coor, last_PI_coor, D_coor_d.begin(), RowSum_2d());

// Calculate the hood using the hood_func
thrust::transform(D_coor_d.begin(), D_coor_d.end(),hood_vec_d.begin(), hood_func(nei_size));


//Copy calculated hood in the cache
thrust::copy(hood_vec_d.begin(),hood_vec_d.end(),nei_cache.begin() + (new_n * c_index));

}
//End of Gamma Function

//New M Matrix Calculation
//Delta * Hood stores in Delta
thrust::transform(D_vector_d.begin(), D_vector_d.begin() + (N * 1), hood_vec_d.begin(), D_vector_d.begin(), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 1), D_vector_d.begin() + (N * 2), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 1), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 2), D_vector_d.begin() + (N * 3), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 2), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 3), D_vector_d.begin() + (N * 4), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 3), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 4), D_vector_d.begin() + (N * 5), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 4), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 5), D_vector_d.begin() + (N * 6), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 5), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 6), D_vector_d.begin() + (N * 7), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 6), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 7), D_vector_d.begin() + (N * 8), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 7), thrust::multiplies<float>());


//Constant learning rate eta value = 0.7
thrust::fill(Eta_d.begin(), Eta_d.end(), 0.7);

//Delta * eta stores in Delta
thrust::transform(D_vector_d.begin(), D_vector_d.end(), Eta_d.begin(), D_vector_d.begin(), thrust::multiplies<float>());


//M - Delta stores in M
thrust::transform(M_vector_d.begin(), M_vector_d.end(), D_vector_d.begin(), M_vector_d.begin(), thrust::minus<float>());


//Write to File section the neurons weight if we are in the last iteration
if (epocs == train-1)
  {
   using namespace std;

   //Stop clock
   c_end = std::clock();


   //File Name buffer and formatting
   char filename[] = "epil_GPU_neurons";
   char underScore[] = "_";
   char txt[] = ".txt";
   char x_dim[33];
   char y_dim[33];
   char iter_v[33];

   snprintf(x_dim, sizeof(x_dim), "%d", x_val);
   snprintf(y_dim, sizeof(y_dim), "%d", y_val);
   snprintf(iter_v, sizeof(iter_v), "%d", iters_val);
   time_t t = time(0);   // get time now

   struct tm * now = localtime( & t );

   char time_buffer [120];
   strftime (time_buffer,120,"%X",now);

   strcat(filename,underScore);
   strcat(filename,x_dim);
   strcat(filename,underScore);
   strcat(filename,y_dim);
   strcat(filename,underScore);
   strcat(filename,iter_v);
   strcat(filename,underScore);
   strcat(filename, time_buffer);
   strcat(filename,underScore);
   std::cout << filename << std::endl;
   strcat(filename,txt);
   std::cout<<filename<<std::endl;

   neuronsFile.open(filename);

   //Iterator to writo to file
   Float8Iterator_d first_M1= thrust::make_zip_iterator(thrust::make_tuple(M_vector_d.begin(),M_vector_d.begin() + (N * 1),M_vector_d.begin() + (N * 2),M_vector_d.begin() + (N * 3),
			                                                   M_vector_d.begin() + (N * 4),M_vector_d.begin() + (N * 5),M_vector_d.begin() + (N * 6),
									   M_vector_d.begin() + (N * 7)));

   Float8 m_1 = first_M1[0];

   //Loop to write to file
   for (size_t i = 0; i < new_n; i++)
     {
         m_1 = first_M1[i];

        neuronsFile << thrust::get<0>(m_1) << " " << thrust::get<1>(m_1) << " " << thrust::get<2>(m_1) << " " << thrust::get<3>(m_1) << " "
                    << thrust::get<4>(m_1) << " " << thrust::get<5>(m_1) << " " << thrust::get<6>(m_1) << " " << thrust::get<7>(m_1) << "\n";
     }
  }

}//New end of main for lo0p/////////////////////////////////////////////////////

//Close neurons weights file
   neuronsFile.close();


float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";

}
