// wdbc-par-vsom.cu/
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


//This functor implements the update M
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


// We'll use a 10-tuple to store our 10d vector type ( we need to call this 3 times for 30 dimensions)
typedef thrust::tuple<float,float,float,float,float,float,float,float,float,float> Float10;
// This functor implements the RowSum for 4d vectors
struct RowSum_10d : public thrust::unary_function<Float10,float>
{
    __host__ __device__
        float operator()(const Float10& a) const
        {
            return thrust::get<0>(a) + thrust::get<1>(a) +
                   thrust::get<2>(a) + thrust::get<3>(a) +
		   thrust::get<4>(a) + thrust::get<5>(a) +
		   thrust::get<6>(a) + thrust::get<7>(a) +
		   thrust::get<8>(a) + thrust::get<9>(a) ;

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
thrust::host_vector<float> random_vector(const size_t N, unsigned int seed = thrust::default_random_engine::default_seed)
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



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Init CLocks
std::clock_t c_start = std::clock();
std::clock_t c_end ;


//Define device vector to hold training data (30 columns)
thrust::device_vector < float > file_d_0;

//Define device iterators to create tuple ( 10 columns matrix)
typedef thrust::device_vector<float>::iterator                  FloatIterator_d;
typedef thrust::
tuple<FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d, FloatIterator_d,FloatIterator_d, FloatIterator_d >  FloatIteratorTuple_d;

typedef thrust::zip_iterator<FloatIteratorTuple_d>            Float10Iterator_d;


srand((unsigned)time(NULL));

//Vectors declaration


//Vector declaration for big vectors to enclose entire matrix:

//Trainig vector D (l x d)
thrust::device_vector < float > file_vector_d;

//Neuron vector M (n x d)
thrust::device_vector<float> M_vector_d = random_vector(N*30,rand() % 100000 + 1);

//Vector X (Holds the selected training instance)
 //  'X' vector N*8=size, and init to 0
thrust::device_vector<float> X_vector_d(N*30,0);
//  'X' vector N*8=size, and init to 0
thrust::host_vector<float> X_vector_h(N*30,0);

//Vector Delta 'D'
// 'D' vector 'Delta' N*8=size, and init to 0
thrust::device_vector<float> D_vector_d(N*30,0);

//Vector PI 'P'
// vector 'P' vector N*8=size, and init to 0
thrust::device_vector<float> P_vector_d(N*30,0);

// n_0 components of the 'S' vector N=size, and init to 0
thrust::device_vector<float> S(N,0);
// n_0..9 components of the 'S' vector N=size, and init to 0
thrust::device_vector<float> S_1(N,0);
  // n_10..19 components of the 'S' vector N=size, and init to 0
thrust::device_vector<float> S_2(N,0);
// n_20..29 components of the 'S' vector N=size, and init to 0
thrust::device_vector<float> S_3(N,0);

//Vector Pc 'P_coordinates' for gamma function
//vector Pc N*2, coordinates for BMU x and y in one vector
thrust::device_vector<float> P_coors(N*2,0);

//vector is a merger of all coordinate x and y  vector N*2 =size, and init to 0
thrust::device_vector<float> P_coors_all(N*2,0);

//Vector PI in coordinates (gamma)
//vector to contain the PI coordinates
thrust::device_vector<float> PI_coors(N*2,0); //vector to contain the PI coordinates

//Create D_coor_d big vector (holds all the distances)
// n_0 components of the 'D_coor' vector N=size, and init to 0
thrust::device_vector<float> D_coor_d(N,0);

//Vector hood to  nei (gamma)
//vector to contain the nei (1,0)
thrust::device_vector<float> hood_vec_d(N,0);

//vector to contain the nei (1,0)
thrust::device_vector<float> hood_vec_d_dim(N*30,0);

//Mask vector in GPU
thrust::device_vector<float> mask_vec_d(N,0);

//Nei cache (Huge vector containing the nei for each instance)
thrust::device_vector<float> nei_cache(new_n * N,0);


//Vector for Eta Valuesr
thrust::device_vector<float> Eta_d(N*30,0);
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
typedef thrust::device_vector<float>::iterator                  FloatIterator_c;
typedef thrust::tuple<FloatIterator_c, FloatIterator_c>    FloatIteratorTuple_c;
typedef thrust::zip_iterator<FloatIteratorTuple_c>             Float2Iterator_c;

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

//11////Create PI_coor Matrix ///////////////////////////////////////////////////////////////////////////////////
// n_0 components of the 'Pi_coor' vector N=size, and init to 0
thrust::device_vector<float> P0_coor(N,0);
 // n_1 components of the 'Pi_coor' vector N=size, and init to 0
thrust::device_vector<float> P1_coor(N,0);

Float2Iterator_c first_P_coor  = thrust::make_zip_iterator(thrust::make_tuple(P0_coor.begin(),P1_coor.begin()));
Float2Iterator_c last_P_coor   = thrust::make_zip_iterator(thrust::make_tuple(P0_coor.end(),P1_coor.end()));

//Create D_coor Matrix ///////////////////////////////////////////////////////////////////////////////////
// n_0 components of the 'D_coor' vector N=size, and init to 0
thrust::device_vector<float> D_coor(N,0);

//Create hood vector
// n_0 components of the 'D_coor' vector N=size, and init to 0
thrust::device_vector<float> hood_vec(N,0);

//Iterator of M Matrix (Random Neuron Weights) for debug
Float10Iterator_d first_M1= thrust::make_zip_iterator(thrust::make_tuple(M_vector_d.begin(), M_vector_d.begin() + (N * 1), M_vector_d.begin() + (N * 2), M_vector_d.begin() + (N*3),
                                                    M_vector_d.begin() + (N*4), M_vector_d.begin() + (N*5), M_vector_d.begin() + (N*6), M_vector_d.begin() + (N*7),
						    M_vector_d.begin() + (N*8),M_vector_d.begin() + (N*9)));
Float10 m_1 = first_M1[0];

//std::cout << "Print out M Matrix (Random Weights)" << std::endl;
//for(size_t i = 0; i < 10; i++)
//  {
//          m_1 = first_M1[i];
//          std::cout << "(" << thrust::get<0>(m_1) << "," << thrust::get<1>(m_1) << "," << thrust::get<2>(m_1) << "," << thrust::get<3>(m_1) << "," << thrust::get<4>(m_1) << ","
//		    << thrust::get<5>(m_1) << "," << thrust::get<6>(m_1) << "," << thrust::get<7>(m_1) << "," << thrust::get<8>(m_1) << "," << thrust::get<9>(m_1) << ")"  << std::endl;
//    }





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Init CLock
//std::clock_t c_start = std::clock();


//Random variable for Xk (use size of dataset + 1)
int v1 = rand() % 99 + 1;

//Initialize the Matrix size to read Matrix D Data
float file_mat[100][30];

//read data file and load into array
    std::ifstream reader("wdbc_norm.txt");
    if (!reader)
    std::cerr << "Error opening file";
    else
    {
       for (int i = 0; i < 100; i++)
       {  for (int j = 0; j < 30; j++)
         {
              reader >> file_mat[i][j];
              reader.ignore();
         }


       }


     }

     //load data into huge vector
       for ( int j = 0; j < 30; j++)
       {  for (int i = 0; i < 100; i++)
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


Float10Iterator_d vector_first_D_1 = thrust::make_zip_iterator(thrust::make_tuple(P_vector_d.begin(),P_vector_d.begin() + (N * 1),P_vector_d.begin() + (N * 2),P_vector_d.begin() + (N*3),
                                  P_vector_d.begin() + (N*4),P_vector_d.begin() + (N*5),P_vector_d.begin() + (N*6),P_vector_d.begin() + (N*7), P_vector_d.begin() +(N*8), P_vector_d.begin() + (N*9)));

Float10Iterator_d vector_first_D_2 = thrust::make_zip_iterator(thrust::make_tuple(P_vector_d.begin()+( N * 10),P_vector_d.begin() + (N * 11),P_vector_d.begin() + (N * 12),P_vector_d.begin() + (N*13),
                                  P_vector_d.begin() + (N*14),P_vector_d.begin() + (N*15),P_vector_d.begin() + (N*16),P_vector_d.begin() + (N*17), P_vector_d.begin() +(N*18), P_vector_d.begin() + (N*19)));

Float10Iterator_d vector_first_D_3 = thrust::make_zip_iterator(thrust::make_tuple(P_vector_d.begin()+( N * 20),P_vector_d.begin() + (N * 21),P_vector_d.begin() + (N * 22),P_vector_d.begin() + (N*23),
                                  P_vector_d.begin() + (N*24),P_vector_d.begin() + (N*25),P_vector_d.begin() + (N*26),P_vector_d.begin() + (N*27), P_vector_d.begin() +(N*28), P_vector_d.begin() + (N*29)));


Float10Iterator_d vector_last_D_1  = thrust::make_zip_iterator(thrust::make_tuple(P_vector_d.begin() + (N * 1), P_vector_d.begin() + (N * 2) ,P_vector_d.begin() + (N *3),P_vector_d.begin() + (N*4),
                                  M_vector_d.begin() + (N*5),M_vector_d.begin() + (N*6),M_vector_d.begin() + (N*7),M_vector_d.begin() + (N*8),M_vector_d.begin() + (N*9),M_vector_d.begin() + (N*10)));


Float10Iterator_d vector_last_D_2  = thrust::make_zip_iterator(thrust::make_tuple(P_vector_d.begin() + (N * 11), P_vector_d.begin() + (N * 12) ,P_vector_d.begin() + (N *13),P_vector_d.begin() + (N*14),
                                  M_vector_d.begin() + (N*15),M_vector_d.begin() + (N*16),M_vector_d.begin() + (N*17),M_vector_d.begin() + (N*18),M_vector_d.begin() + (N*19),M_vector_d.begin() + (N*20)));


Float10Iterator_d vector_last_D_3  = thrust::make_zip_iterator(thrust::make_tuple(P_vector_d.begin() + (N * 21), P_vector_d.begin() + (N * 22) ,P_vector_d.begin() + (N *23),P_vector_d.begin() + (N*24),
                                  M_vector_d.begin() + (N*25),M_vector_d.begin() + (N*26),M_vector_d.begin() + (N*27),M_vector_d.begin() + (N*28),M_vector_d.begin() + (N*29),M_vector_d.begin() + (N*30)));


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

v1 = rand() % 99 + 1;


//This creates the D vector Repeting the raw_pointer value in the range specificy by X_vector (saves repetion  value in D_vector)

//D1
thrust::transform(X_vector_d.begin(), X_vector_d.begin()+(N * 1), X_vector_d.begin(),X_vector_d.begin(), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[v1]))));
//D2
thrust::transform(X_vector_d.begin() + ( N * 1), X_vector_d.begin()+(N * 2), X_vector_d.begin() +( N * 1),X_vector_d.begin()+(N * 1), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 1 + v1]))));
//D3
thrust::transform(X_vector_d.begin()+ ( N * 2), X_vector_d.begin()+(N * 3), X_vector_d.begin() + ( N * 2) ,X_vector_d.begin()+(N * 2), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 2 + v1]))));
//D4
thrust::transform(X_vector_d.begin() +( N * 3), X_vector_d.begin()+(N * 4), X_vector_d.begin() +(N * 3),X_vector_d.begin()+(N * 3), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 3 + v1]))));
//D5
thrust::transform(X_vector_d.begin() +( N * 4), X_vector_d.begin()+(N * 5), X_vector_d.begin() +(N * 4),X_vector_d.begin()+(N * 4), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 4 + v1]))));
//D6
thrust::transform(X_vector_d.begin() +( N * 5), X_vector_d.begin()+(N * 6), X_vector_d.begin() +(N * 5),X_vector_d.begin()+(N * 5), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 5 + v1]))));
//D7
thrust::transform(X_vector_d.begin() +( N * 6), X_vector_d.begin()+(N * 7), X_vector_d.begin() +(N * 6),X_vector_d.begin()+(N * 6), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 6 + v1]))));
//D8
thrust::transform(X_vector_d.begin() +( N * 7), X_vector_d.begin()+(N * 8), X_vector_d.begin() +(N * 7),X_vector_d.begin()+(N * 7), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 7 + v1]))));

//D9
thrust::transform(X_vector_d.begin() + (N * 8), X_vector_d.begin()+(N * 9), X_vector_d.begin() + (N * 8), X_vector_d.begin()+(N *8), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 8 + v1]))));
//D10
thrust::transform(X_vector_d.begin() + ( N * 9), X_vector_d.begin()+(N * 10), X_vector_d.begin() +(N * 9),X_vector_d.begin()+(N * 9), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 9 + v1]))));
//D11
thrust::transform(X_vector_d.begin()+ ( N * 10), X_vector_d.begin()+(N * 11), X_vector_d.begin() +(N * 10) ,X_vector_d.begin()+(N * 10), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 10 + v1]))));
//D12
thrust::transform(X_vector_d.begin() +( N * 11), X_vector_d.begin()+(N * 12), X_vector_d.begin() +(N * 11),X_vector_d.begin()+(N * 11), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 11 + v1]))));
//D13
thrust::transform(X_vector_d.begin() +( N * 12), X_vector_d.begin()+(N * 13), X_vector_d.begin() +(N * 12),X_vector_d.begin()+(N * 12), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 12 + v1]))));
//D14
thrust::transform(X_vector_d.begin() +( N * 13), X_vector_d.begin()+(N * 14), X_vector_d.begin() +(N * 13),X_vector_d.begin()+(N * 13), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 13 + v1]))));
//D15
thrust::transform(X_vector_d.begin() +( N * 14), X_vector_d.begin()+(N * 15), X_vector_d.begin() +(N * 14),X_vector_d.begin()+(N * 14), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 14 + v1]))));
//D16
thrust::transform(X_vector_d.begin() +( N * 15), X_vector_d.begin()+(N * 16), X_vector_d.begin() +(N * 15),X_vector_d.begin()+(N * 15), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 15 + v1]))));

//D17
thrust::transform(X_vector_d.begin() + (N * 16), X_vector_d.begin()+(N * 17), X_vector_d.begin() + (N * 16), X_vector_d.begin()+( N*16), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 16 + v1]))));
//D18
thrust::transform(X_vector_d.begin() + ( N * 17), X_vector_d.begin()+(N * 18), X_vector_d.begin() +(N * 17),X_vector_d.begin()+(N * 17), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 17 + v1]))));
//D19
thrust::transform(X_vector_d.begin()+ ( N * 18), X_vector_d.begin()+(N * 19), X_vector_d.begin() +(N * 18) ,X_vector_d.begin()+(N * 18), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 18 + v1]))));
//D20
thrust::transform(X_vector_d.begin() +( N * 19), X_vector_d.begin()+(N * 20), X_vector_d.begin() +(N * 19),X_vector_d.begin()+(N * 19), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 19 + v1]))));
//D21
thrust::transform(X_vector_d.begin() +( N * 20), X_vector_d.begin()+(N * 21), X_vector_d.begin() +(N * 20),X_vector_d.begin()+(N * 20), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 20 + v1]))));
//D22
thrust::transform(X_vector_d.begin() +( N * 21), X_vector_d.begin()+(N * 22), X_vector_d.begin() +(N * 21),X_vector_d.begin()+(N * 21), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 21 + v1]))));
//D23
thrust::transform(X_vector_d.begin() +( N * 22), X_vector_d.begin()+(N * 23), X_vector_d.begin() +(N * 22),X_vector_d.begin()+(N * 22), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 22 + v1]))));
//D24
thrust::transform(X_vector_d.begin() +( N * 23), X_vector_d.begin()+(N * 24), X_vector_d.begin() +(N * 23),X_vector_d.begin()+(N * 23), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 23 + v1]))));


//D25
thrust::transform(X_vector_d.begin() + (N * 24), X_vector_d.begin()+(N * 25), X_vector_d.begin() + (N * 24), X_vector_d.begin()+( N *24), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 24 + v1]))));
//D26
thrust::transform(X_vector_d.begin() + ( N * 25), X_vector_d.begin()+(N * 26), X_vector_d.begin() +(N * 25),X_vector_d.begin()+(N * 25), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 25 + v1]))));
//D27
thrust::transform(X_vector_d.begin()+ ( N * 26), X_vector_d.begin()+(N * 27), X_vector_d.begin() +(N * 26) ,X_vector_d.begin()+(N * 26), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 26 + v1]))));
//D28
thrust::transform(X_vector_d.begin() +( N * 27), X_vector_d.begin()+(N * 28), X_vector_d.begin() +(N * 27),X_vector_d.begin()+(N * 27), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 27 + v1]))));
//D29
thrust::transform(X_vector_d.begin() +( N * 28), X_vector_d.begin()+(N * 29), X_vector_d.begin() +(N * 28),X_vector_d.begin()+(N * 28), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 28 + v1]))));
//D30
thrust::transform(X_vector_d.begin() +( N * 29), X_vector_d.begin()+(N * 30), X_vector_d.begin() +(N * 29),X_vector_d.begin()+(N * 29), saxpy_functor(thrust::raw_pointer_cast(&(file_vector_d[100 * 29 + v1]))));


//Calculate delta vector :
thrust::transform(M_vector_d.begin(), M_vector_d.end(), X_vector_d.begin(), D_vector_d.begin(), thrust::minus<float>());

//Calculate PI vector:
thrust::transform(D_vector_d.begin(), D_vector_d.end(), D_vector_d.begin(), P_vector_d.begin(), thrust::multiplies<float>());


//Calculate S Vector with Rowsum reduction and store in S vector
//thrust::transform(vector_first_D, vector_last_D, S.begin(), RowSum_8d());

thrust::transform(vector_first_D_1, vector_last_D_1, S_1.begin(), RowSum_10d());
thrust::transform(vector_first_D_2, vector_last_D_2, S_2.begin(), RowSum_10d());
thrust::transform(vector_first_D_3, vector_last_D_3, S_3.begin(), RowSum_10d());

//Collapse all the S (10 dims each) vectors sections into one S vector
thrust::transform(S_1.begin(), S_1.end(),S_2.begin(),S.begin(),thrust::plus<float>());
thrust::transform(S_3.begin(), S_3.end(),S.begin(),S.begin(),thrust::plus<float>());
//thrust::transform(vector_first_D_1, vector_last_D_1, S.begin(), RowSum_10d());

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


//Start Update neighborhood //////////////////////////////////////////////////////////////////////////
//Gamma Function /////


//Init PC x
x = P_coor0[c_index];

//Init PC y
y = P_coor1[c_index];


//Fill Matrix C with index coordinates X <-- 1^m * Pc
thrust::fill(P_coors.begin(), P_coors.begin()+ N, x);
thrust::fill(P_coors.begin()+ N, P_coors.end(), y);


//Apply the find coor nei  transformation to  big coodinates vector (vector with X and Y)
thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(P_coors_all.begin(),P_coors.begin(),PI_coors.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(P_coors_all.end(),P_coors.end(),PI_coors.end())),
                     find_nei_functor());

// Finally, we pass the zip_iterators into transform() as if they
// were 'normal' iterators for a device_vector<Float2>.
thrust::transform(first_PI_coor, last_PI_coor, D_coor_d.begin(), RowSum_2d());

// Calculate the hood using the hood_func
thrust::transform(D_coor_d.begin(), D_coor_d.end(),hood_vec_d.begin(), hood_func(nei_size));

//copy calculated hood in the cache
thrust::copy(hood_vec_d.begin(),hood_vec_d.end(),nei_cache.begin() + (new_n * c_index));

}

//New M Matrix Calculation
//Delta * Hood stores in Delta

// The folloing lines can be implemented inside a for loop
//e.g : for (i = 1 ; i < 31; i++)
//thrust::transform(D_vector_d.begin() + ( N * i), D_vector_d.begin() + (N * (i + 1)), hood_vec_d.begin(), D_vector_d.begin()+ ( N * i), thrust::multiplies<float>());
// The sequential code allow for easier debuggin and does not impact the timming

thrust::transform(D_vector_d.begin(), D_vector_d.begin() + (N * 1), hood_vec_d.begin(), D_vector_d.begin(), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 1), D_vector_d.begin() + (N * 2), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 1), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 2), D_vector_d.begin() + (N * 3), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 2), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 3), D_vector_d.begin() + (N * 4), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 3), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 4), D_vector_d.begin() + (N * 5), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 4), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 5), D_vector_d.begin() + (N * 6), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 5), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 6), D_vector_d.begin() + (N * 7), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 6), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 7), D_vector_d.begin() + (N * 8), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 7), thrust::multiplies<float>());


thrust::transform(D_vector_d.begin() + ( N * 8), D_vector_d.begin() + (N * 9), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 8), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 9), D_vector_d.begin() + (N * 10), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 9), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 10), D_vector_d.begin() + (N * 11), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 10), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 11), D_vector_d.begin() + (N * 12), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 11), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 12), D_vector_d.begin() + (N * 13), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 12), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 13), D_vector_d.begin() + (N * 14), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 13), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 14), D_vector_d.begin() + (N * 15), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 14), thrust::multiplies<float>());


thrust::transform(D_vector_d.begin() + ( N * 15), D_vector_d.begin() + (N * 16), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 15), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 16), D_vector_d.begin() + (N * 17), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 16), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 17), D_vector_d.begin() + (N * 18), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 17), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 18), D_vector_d.begin() + (N * 19), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 18), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 19), D_vector_d.begin() + (N * 20), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 19), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 20), D_vector_d.begin() + (N * 21), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 20), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 21), D_vector_d.begin() + (N * 22), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 21), thrust::multiplies<float>());


thrust::transform(D_vector_d.begin() + ( N * 22), D_vector_d.begin() + (N * 23), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 22), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 23), D_vector_d.begin() + (N * 24), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 23), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 24), D_vector_d.begin() + (N * 25), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 24), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 25), D_vector_d.begin() + (N * 26), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 25), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 26), D_vector_d.begin() + (N * 27), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 26), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 27), D_vector_d.begin() + (N * 28), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 27), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 28), D_vector_d.begin() + (N * 29), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 28), thrust::multiplies<float>());
thrust::transform(D_vector_d.begin() + ( N * 29), D_vector_d.begin() + (N * 30), hood_vec_d.begin(), D_vector_d.begin()+ ( N * 29), thrust::multiplies<float>());



//Eta Constant
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
   char filename[] = "wdbc_GPU_neurons_norm";
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
   Float10Iterator_d first_M1= thrust::make_zip_iterator(thrust::make_tuple(M_vector_d.begin(),M_vector_d.begin() + (N * 1),M_vector_d.begin() + (N * 2),M_vector_d.begin() + (N * 3),
			                                                   M_vector_d.begin() + (N * 4),M_vector_d.begin() + (N * 5),M_vector_d.begin() + (N * 6),
									   M_vector_d.begin() + (N * 7), M_vector_d.begin() + (N * 8),M_vector_d.begin() + (N * 9)));

   Float10Iterator_d first_M2= thrust::make_zip_iterator(thrust::make_tuple(M_vector_d.begin()+ ( N * 10) , M_vector_d.begin() + (N * 11),M_vector_d.begin() + (N * 12),M_vector_d.begin() + (N * 13),
                                                                           M_vector_d.begin() + (N * 14),M_vector_d.begin() + (N * 15),M_vector_d.begin() + (N * 16),
                                                                           M_vector_d.begin() + (N * 17), M_vector_d.begin() + (N * 18),M_vector_d.begin() + (N * 19)));

   Float10Iterator_d first_M3= thrust::make_zip_iterator(thrust::make_tuple(M_vector_d.begin()+ ( N * 20) , M_vector_d.begin() + (N * 21),M_vector_d.begin() + (N * 22),M_vector_d.begin() + (N * 23),
                                                                           M_vector_d.begin() + (N * 24),M_vector_d.begin() + (N * 25),M_vector_d.begin() + (N * 26),
                                                                           M_vector_d.begin() + (N * 27), M_vector_d.begin() + (N * 28),M_vector_d.begin() + (N * 29)));



   Float10 m_1 = first_M1[0];
   Float10 m_2 = first_M2[0];
   Float10 m_3 = first_M3[0];

   //Loop to write to file
   for (size_t i = 0; i < new_n; i++)
     {

         m_1 = first_M1[i];
	       m_2 = first_M2[i];
	       m_3 = first_M3[i];


        neuronsFile << thrust::get<0>(m_1) << " " << thrust::get<1>(m_1) << " " << thrust::get<2>(m_1) << " " << thrust::get<3>(m_1) << " " << thrust::get<4>(m_1) << " "
	            << thrust::get<5>(m_1) << " " << thrust::get<6>(m_1) << " " << thrust::get<7>(m_1) << " " << thrust::get<8>(m_1) << " " << thrust::get<9>(m_1) << " "
		    << thrust::get<0>(m_2) << " " << thrust::get<1>(m_2) << " " << thrust::get<2>(m_2) << " " << thrust::get<3>(m_2) << " " << thrust::get<4>(m_2) << " "
                    << thrust::get<5>(m_2) << " " << thrust::get<6>(m_2) << " " << thrust::get<7>(m_2) << " " << thrust::get<8>(m_2) << " " << thrust::get<9>(m_2) << " "
		    << thrust::get<0>(m_3) << " " << thrust::get<1>(m_3) << " " << thrust::get<2>(m_3) << " " << thrust::get<3>(m_3) << " " << thrust::get<4>(m_3) << " "
                    << thrust::get<5>(m_3) << " " << thrust::get<6>(m_3) << " " << thrust::get<7>(m_3) << " " << thrust::get<8>(m_3) << " " << thrust::get<9>(m_3) << "\n";

     }
  }

}//New end of main for
   neuronsFile.close();

float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";


}
