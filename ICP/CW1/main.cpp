#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <nanogui/formhelper.h>
#include <igl/fit_plane.h>
#include <nanogui/screen.h>
#include "tutorial_shared_path.h"
#include "nanoflann.hpp"
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <random>

using namespace Eigen;
using namespace std;
using namespace nanoflann;

typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, Dynamic, Dynamic> >  my_kd_tree_t;

Eigen::MatrixXd V1, V2,samples,M1,M2,M3,M4,M5;
Eigen::MatrixXi F1, F2,Fm1,Fm2,Fm3,Fm4,Fm5;

#define PI 3.14159265

// number of query result
const size_t num_results = 1;
// epsilon value for comparing small number
const double epsilon = 1e-8;


void show_meshes(igl::viewer::Viewer& viewer,Eigen::MatrixXd P,Eigen::MatrixXd Q,Eigen::MatrixXi Fp,Eigen::MatrixXi Fq){

    // concatenate vertices data in two meshes.
    Eigen::MatrixXd V(P.rows() + Q.rows(), P.cols());
    V << P, Q;
    // concatenate faces, F2 start from index number of elements in F1.
    Eigen::MatrixXi F(Fp.rows() + Fq.rows(), Fp.cols());
    F << Fp, (Fq.array() + P.rows());

    // assign different color to two groups of meshes.
    Eigen::MatrixXd C(F.rows(), 3);
    C << Eigen::RowVector3d(1.0, 0.0, 0.0).replicate(Fp.rows(), 1),
            Eigen::RowVector3d(0.0, 1.0, 0.0).replicate(Fq.rows(), 1);

    viewer.data.clear();
    viewer.data.set_mesh(V, F);
    viewer.data.set_colors(C);
    viewer.core.align_camera_center(V,F);
}

void show_multiple_meshes(igl::viewer::Viewer& viewer){

    Eigen::MatrixXd V(M1.rows()+M2.rows()+M3.rows()+M4.rows()+M5.rows(), M1.cols());
    V << M1,M2,M3,M4,M5;

    Eigen::MatrixXi F(Fm1.rows()+Fm2.rows()+Fm3.rows()+Fm4.rows()+Fm5.rows(), Fm1.cols());
    F << Fm1,
            (Fm2.array() + M1.rows()),
            (Fm3.array() + M1.rows()+M2.rows()),
            (Fm4.array() + M1.rows()+M2.rows()+M3.rows()),
            (Fm5.array() + M1.rows()+M2.rows()+M3.rows()+M4.rows());

    Eigen::MatrixXd C(F.rows(), 3);
    C << Eigen::RowVector3d(1.0, 0.0, 0.0).replicate(Fm1.rows(), 1),
            Eigen::RowVector3d(0.0, 1.0, 0.0).replicate(Fm2.rows(), 1),
            Eigen::RowVector3d(0.0, 0.0, 1.0).replicate(Fm3.rows(), 1),
            Eigen::RowVector3d(1.0, 1.0, 1.0).replicate(Fm4.rows(), 1),
            Eigen::RowVector3d(0.0, 0.0, 0.0).replicate(Fm5.rows(), 1);

    viewer.data.clear();
    viewer.data.set_mesh(V, F);
    viewer.data.set_colors(C);
    viewer.core.align_camera_center(V,F);
}


// Apply the rotation and translation we estimated from the ICP algorithm to the point cloud
Eigen::MatrixXd apply_rigid_transform(Eigen::MatrixXd P, Eigen::MatrixXd R, Eigen::MatrixXd t){

    for (size_t i = 0; i < P.rows(); i++)
        P.row(i) = (R * P.row(i).transpose() + t).transpose();

    return P;
}

// Apply the rotation about different axis
Eigen::MatrixXd apply_rotation(Eigen::MatrixXd P,float a_x,float a_y, float a_z){

    Eigen::MatrixXd R_x = Eigen::MatrixXd::Zero(3,3);
    Eigen::MatrixXd R_y = Eigen::MatrixXd::Zero(3,3);
    Eigen::MatrixXd R_z = Eigen::MatrixXd::Zero(3,3);

    // convert angle into radian
    a_x = a_x * PI / 180.0;
    a_y = a_y * PI / 180.0;
    a_z = a_z * PI / 180.0;

    // construct different rotation matrix
    R_x<<   1.0,0.0,0.0,
            0.0,cos(a_x),-sin(a_x),
            0.0,sin(a_x),cos(a_x);

    R_y<<   cos(a_y),0.0,sin(a_y),
            0.0,1.0,0.0,
            -sin(a_y),0.0,cos(a_y);

    R_z<<   cos(a_z),-sin(a_z),0.0,
            sin(a_z),cos(a_z),0.0,
            0.0,0.0,1.0;

    P*=R_x * R_y * R_z;

    return P;
}
Eigen::MatrixXd apply_noise(Eigen::MatrixXd P,float noise_lv){

    // compute the bounding dimension of the point cloud
    Eigen::Vector3d min = P.colwise().minCoeff();
    Eigen::Vector3d max = P.colwise().maxCoeff();

    // compute a zero mean noise with given noise level
    Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(P.rows(),P.cols());
    noise.col(0) = noise_lv * (max(0)-min(0)) * (Eigen::MatrixXd::Random(P.rows(),1)- .5 * Eigen::MatrixXd::Ones(P.rows(),1));
    noise.col(1) = noise_lv * (max(1)-min(2)) * (Eigen::MatrixXd::Random(P.rows(),1)- .5 * Eigen::MatrixXd::Ones(P.rows(),1));
    noise.col(2) = noise_lv * (max(1)-min(2)) * (Eigen::MatrixXd::Random(P.rows(),1)- .5 * Eigen::MatrixXd::Ones(P.rows(),1));

    return P+= noise;
}

Eigen::MatrixXd uniform_sampling(Eigen::MatrixXd P,float sampling_rate){

    // if the sampling rate is larger than 1, than just return the raw data set
    if(sampling_rate>=1.0)
        return P;

    int num_sample = int(sampling_rate * P.rows());

    MatrixXd samp =Eigen::MatrixXd::Zero(num_sample,P.cols());

    // creates a random sampling engine
    default_random_engine e;
    uniform_int_distribution<unsigned> u(0, P.rows()-1);

    // Selcet uniform distributed random number beween 0 to number of sample points
    for(int i=0; i<num_sample; i++)
        samp.row(i) = P.row(u(e));

    return samp;
}
// calculates the local point to point error
double calculate_p2p_error(Eigen::MatrixXd P,Eigen::MatrixXd Q){
    double E = 0.0;
    Eigen::MatrixXd pi, qi;

    for (size_t idx = 0; idx < Q.rows(); idx++){
        pi = P.row(idx);
        qi = Q.row(idx);
        // calculate point to point L2 distance of given correspondence mapping
        E+= pow(pi(0)-qi(0),2.0)+pow(pi(1)-qi(1),2.0)+pow(pi(1)-qi(1),2.0);
    }

    return E;
}
// calculates the point to plane error of two point set with normal
double calculate_point_to_plane_error(Eigen::MatrixXd P,Eigen::MatrixXd Q,Eigen::MatrixXd Np){
    double E = 0.0;
    Eigen::Vector3d ni;
    Eigen::Vector3d diff;

    for(size_t idx =0;idx<Q.rows();idx++){
        ni = Np.row(idx);
        diff = Q.row(idx)-P.row(idx);
        E+= pow(diff.dot(ni),2.0);
    }

    return E;
}

// calculates the total point to point error
double calculate_total_p2p_error(Eigen::MatrixXd P,Eigen::MatrixXd Q){
    double E = 0.0;
    Eigen::MatrixXd pi, qi;

    // Build KD-tree for all vertices in P
    size_t nSamples = P.rows();

    Eigen::Matrix<double, Dynamic, Dynamic>  mat(nSamples, 3);

    for (size_t i = 0; i < P.rows(); i++)
        for (size_t d = 0; d < 3; d++)
            mat(i, d) = P(i, d);

    // Set the depth of the kd-three to be 10
    my_kd_tree_t   mat_index(mat, 10 /* max leaf */ );
    mat_index.index->buildIndex();

    for (size_t idx = 0; idx < Q.rows(); idx++){
        qi = Q.row(idx);

        //Build the point in Q that we want to query its closest point in P
        std::vector<double> query_pt(3);
        for (size_t d = 0; d < 3; d++)
            query_pt[d] = Q(idx, d);

        // Init data set for storing the index of the closest point and its square distance
        vector<size_t> ret_indexes(num_results);
        vector<double> out_dists_sqr(num_results);

        // Init the result set for query
        nanoflann::KNNResultSet<double> resultSet(num_results);
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

        // Perform find nearest neighbor query
        mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        pi = P.row(ret_indexes[0]);

        // sum up the point to point L2 of given data point clouds
        E+= pow(pi(0)-qi(0),2.0)+pow(pi(1)-qi(1),2.0)+pow(pi(1)-qi(1),2.0);
    }

    return E;
}

Eigen::MatrixXd estimateNormal(Eigen::MatrixXd P){

    Eigen::MatrixXd Np = Eigen::MatrixXd::Zero(P.rows(),P.cols());

    size_t nSamples = P.rows();
    Eigen::Matrix<double, Dynamic, Dynamic>  mat(nSamples, 3);

    for (size_t i = 0; i < P.rows(); i++)
        for (size_t d = 0; d < 3; d++)
            mat(i,d) = P(i,d);

    my_kd_tree_t   mat_index(mat, 10 /* max leaf */ );
    mat_index.index->buildIndex();

    Eigen::MatrixXd meancenter = P.colwise().sum() / double(P.rows());

    for (int idx =0; idx<P.rows(); idx++){
        std::vector<double> query_pt(3);
        for (size_t d = 0; d < 3; d++)
            query_pt[d] = P(idx,d);

        const size_t num_results = 8;
        vector<size_t>   ret_indexes(num_results);
        vector<double> out_dists_sqr(num_results);

        nanoflann::KNNResultSet<double> resultSet(num_results);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
        mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        Eigen::MatrixXd Selectpoints(num_results,3);
        for (size_t i=0; i<num_results; i++) {
            Selectpoints(i,0) = P(ret_indexes[i],0);
            Selectpoints(i,1) = P(ret_indexes[i],1);
            Selectpoints(i,2) = P(ret_indexes[i],2);
        }

        Eigen::RowVector3d Npt,Ct;

        igl::fit_plane(Selectpoints,Npt,Ct);

        Np(idx,0) = Npt(0);
        Np(idx,1) = Npt(1);
        Np(idx,2) = Npt(2);

        if ((meancenter(0,0)-P(idx,0)) * Np(idx,0) + (meancenter(0,1)-P(idx,1)) * Np(idx,1) + (meancenter(0,2)-P(idx,2)) * Np(idx,2) > 0) {
            Np(idx,0) = -Npt(0);
            Np(idx,1) = -Npt(1);
            Np(idx,2) = -Npt(2);
        }

    }

    return Np;
}

void performPoint2PlaneICP(Eigen::MatrixXd P,Eigen::MatrixXd& Q, int max_iteration){
    Eigen::MatrixXd Np = estimateNormal(P);

    // Build KD-tree for all vertices in P
    size_t nSamples = P.rows();

    // if sampling is turn on, then compute the sampling set, otherwise uses the raw data set as samples
    samples = Q;

    Eigen::Matrix<double, Dynamic, Dynamic>  mat(nSamples, 3);

    for (size_t i = 0; i < P.rows(); i++)
        for (size_t d = 0; d < 3; d++)
            mat(i, d) = P(i, d);

    // Set the depth of the kd-three to be 10
    my_kd_tree_t   mat_index(mat, 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // Fields for storing Energy in current and last iteration of ICP
    double E =0.0;
    double current_E = 0.0;
    int itr_count = 0;

    do{
        //update Energy of last iteration
        E = current_E;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(samples.rows(), 6);
        Eigen::MatrixXd b = Eigen::MatrixXd::Zero(samples.rows(), 1);

        Eigen::MatrixXd sample_p = Eigen::MatrixXd::Zero(samples.rows(), 3);
        Eigen::MatrixXd sample_n = Eigen::MatrixXd::Zero(samples.rows(), 3);

        for (int idx = 0; idx < samples.rows(); idx++) {

            //Build the point in Q that we want to query its closest point in P
            std::vector<double> query_pt(3);
            for (size_t d = 0; d < 3; d++)
                query_pt[d] = samples(idx, d);

            // Init data set for storing the index of the closest point and its square distance
            vector<size_t> ret_indexes(num_results);
            vector<double> out_dists_sqr(num_results);

            // Init the result set for query
            nanoflann::KNNResultSet<double> resultSet(num_results);
            resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

            // Perform find nearest neighbor query
            mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

            sample_p.row(idx) = P.row(ret_indexes[0]);
            sample_n.row(idx) = Np.row(ret_indexes[0]);

            Eigen::Vector3d pi = sample_p.row(idx);
            Eigen::Vector3d qi = samples.row(idx);
            Eigen::Vector3d ni = sample_n.row(idx);

            Eigen::Vector3d diff = samples.row(idx)-sample_p.row(idx);

            // Building Error matrix A and scalar result b
            A.row(idx) << qi.cross(ni).transpose(),ni.transpose();
            b.row(idx) << -diff.dot(ni);

        }

        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3, 3);
        Eigen::MatrixXd t = Eigen::MatrixXd::Zero(1,3);

        // solve for the least squares
        Eigen::MatrixXd x = A.colPivHouseholderQr().solve(b);;

        // takes the first three column as angle in x,y,z axis, reconstruct the rotation matrix
        R<<cos(x(2))*cos(x(1)),
                -sin(x(2))*cos(x(0))+cos(x(2))*sin(x(1))*sin(x(0)),
                sin(x(2))*sin(x(0))+cos(x(2))*sin(x(1))*cos(x(0)),
                sin(x(2))*cos(x(1)),
                cos(x(2))*cos(x(0))+sin(x(2))*sin(x(1))*sin(x(0)),
                -cos(x(2))*sin(x(0))+sin(x(2))*sin(x(1))*cos(x(0)),
                -sin(x(1)),
                cos(x(1))*sin(x(0)),
                cos(x(1))*cos(x(0));

        // take the last three row as translation
        t<<x(3),x(4),x(5);

        // Apply rigid transformation we estimated to the data
        Q = apply_rigid_transform(Q,R,t);
        samples = apply_rigid_transform(samples,R,t);

        current_E = calculate_point_to_plane_error(sample_p,samples,sample_n);
        // Shows the energy
        itr_count++;
        std::cout<<"P2P ICP iteration "<<itr_count<<", current local energy:"<<current_E<<endl;

    }while(abs(E-current_E)>epsilon && itr_count < max_iteration);
    // terminate when the energy change is nearly 0 or maximum iteration is reached
}



void performICP(Eigen::MatrixXd P,Eigen::MatrixXd& Q, int max_iteration,bool isSampling,float sampling_rate){

    // Build KD-tree for all vertices in P
    size_t nSamples = P.rows();

    // if sampling is turn on, then compute the sampling set, otherwise uses the raw data set as samples
    samples = isSampling?uniform_sampling(Q,sampling_rate):Q;

    Eigen::Matrix<double, Dynamic, Dynamic>  mat(nSamples, 3);

    for (size_t i = 0; i < P.rows(); i++)
        for (size_t d = 0; d < 3; d++)
            mat(i, d) = P(i, d);

    // Set the depth of the kd-three to be 10
    my_kd_tree_t   mat_index(mat, 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // Fields for storing Energy in current and last iteration of ICP
    double E =0.0;
    double current_E = 0.0;
    int itr_count = 0;

    do {
        //update Energy of last iteration
        E = current_E;

        // Init Matrix A for storing Sum q_i*p_i'
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);

        // Calculates the mean centre point for each point set
        Eigen::MatrixXd mean_p, mean_q;
        // Init the Matrix to store correspondence of p_i
        Eigen::MatrixXd sample_p = Eigen::MatrixXd::Zero(samples.rows(), 3);

        for (int idx = 0; idx < samples.rows(); idx++) {

            //Build the point in Q that we want to query its closest point in P
            std::vector<double> query_pt(3);
            for (size_t d = 0; d < 3; d++)
                query_pt[d] = samples(idx, d);

            // Init data set for storing the index of the closest point and its square distance
            vector<size_t> ret_indexes(num_results);
            vector<double> out_dists_sqr(num_results);

            // Init the result set for query
            nanoflann::KNNResultSet<double> resultSet(num_results);
            resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

            // Perform find nearest neighbor query
            mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

            // Assign the nearest point to the matrix
            sample_p.row(idx) = P.row(ret_indexes[0]);

        }

        // Transpose row vector to col vector for further calculations
        mean_p = sample_p.colwise().sum() / double(sample_p.rows());
        mean_p.transposeInPlace();

        mean_q = samples.colwise().sum() / double(samples.rows());
        mean_q.transposeInPlace();


        for (int j = 0; j < samples.rows(); j++) {
            // Init p_hat and q_hat for storing the bary-centered point
            Eigen::MatrixXd p_hat;
            Eigen::MatrixXd q_hat;

            // Compute the bary-centred point p_hat and q_hat
            p_hat = sample_p.row(j).transpose() - mean_p;
            q_hat = samples.row(j).transpose() - mean_q;

            // Sum up the A matrix
            A += q_hat * p_hat.transpose();
        }

        // Compute the SVD decomposition of A
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        Eigen::MatrixXd R, t;

        // Compute the Rotation where R = V*U'
        R = svd.matrixV() * svd.matrixU().transpose();
        // Compute the translation where t = p_bar - R* q_bar
        t = mean_p - R * mean_q;

        // Update the data set Q with new alignment
        Q = apply_rigid_transform(Q, R, t);
        samples = apply_rigid_transform(samples,R,t);

        // Update the Energy of current Iteration
        current_E = calculate_p2p_error(sample_p,samples);

        // Shows the energy
        itr_count++;
        std::cout<<"P2P ICP iteration "<<itr_count<<", current local energy:"<<current_E<<endl;

    }while(abs(E-current_E)>epsilon && itr_count < max_iteration);
    // terminate when the energy change is nearly 0 or maximum iteration is reached

}

void perform_ICP_with_multiple_scans(Eigen::MatrixXd scans[],int scan_count,float sampling_rate){

    Eigen::MatrixXd P,Q;
    Eigen::MatrixXd Rs[scan_count];
    Eigen::MatrixXd ts[scan_count];

    for (int i =1;i<scan_count;i++){

        Q = scans[i];
        samples = uniform_sampling(Q,sampling_rate);

        int sample_count = samples.rows();
        int count = 0;
        Eigen::MatrixXd PointMates[scan_count-1];

        for (int j=0;j<scan_count;j++){

            if(i==j)
                continue;

            P = scans[j];

            Eigen::MatrixXd sample_p = Eigen::MatrixXd::Zero(sample_count, 3);

            // Build KD-tree for all vertices in P
            size_t nSamples = P.rows();
            Eigen::Matrix<double, Dynamic, Dynamic>  mat(nSamples, 3);

            for (size_t k = 0; k < P.rows(); k++)
                for (size_t d = 0; d < 3; d++)
                    mat(k, d) = P(k, d);

            // Set the depth of the kd-three to be 10
            my_kd_tree_t   mat_index(mat, 10 /* max leaf */ );
            mat_index.index->buildIndex();

            for (int idx = 0; idx < samples.rows(); idx++) {

                //Build the point in Q that we want to query its closest point in P
                std::vector<double> query_pt(3);
                for (size_t d = 0; d < 3; d++)
                    query_pt[d] = samples(idx, d);

                // Init data set for storing the index of the closest point and its square distance
                vector<size_t> ret_indexes(num_results);
                vector<double> out_dists_sqr(num_results);

                // Init the result set for query
                nanoflann::KNNResultSet<double> resultSet(num_results);
                resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

                // Perform find nearest neighbor query
                mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

                sample_p.row(idx) = P.row(ret_indexes[0]);
            }

            PointMates[count] = sample_p;
            count++;

        }

        Eigen::MatrixXd MateP = Eigen::MatrixXd::Zero(sample_count*scan_count-1, 3);
        MateP<<PointMates[0],PointMates[1],PointMates[2],PointMates[3];
        Eigen::MatrixXd MateQ = Eigen::MatrixXd::Zero(sample_count*scan_count-1, 3);
        MateQ<<samples,samples,samples,samples;

        Eigen::MatrixXd mean_p, mean_q;

        mean_p = MateP.colwise().sum() / double(MateP.rows());
        mean_p.transposeInPlace();

        mean_q = MateQ.colwise().sum() / double(MateQ.rows());
        mean_q.transposeInPlace();

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
        Eigen::MatrixXd R, t;

        for (int j = 0; j < MateP.rows(); j++) {
            // Init p_hat and q_hat for storing the bary-centered point
            Eigen::MatrixXd p_hat;
            Eigen::MatrixXd q_hat;

            // Compute the bary-centred point p_hat and q_hat
            p_hat = MateP.row(j).transpose() - mean_p;
            q_hat = MateQ.row(j).transpose() - mean_q;

            // Sum up the A matrix
            A += q_hat * p_hat.transpose();
        }

        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);

        Rs[i] = svd.matrixV() * svd.matrixU().transpose();
        ts[i] = mean_p - Rs[i] * mean_q;
    }

    for (int i = 1;i<scan_count;i++){
        scans[i] = apply_rigid_transform(scans[i], Rs[i], ts[i]);
    }
}


int main(int argc, char *argv[]) {

    // Load meshes in OFF format
    igl::readOFF("../tutorial/CW1/bun000.off", V1, F1);
    igl::readOFF("../tutorial/CW1/bun045.off", V2, F2);

    float a_x = 0.0;
    float a_y = 0.0;
    float a_z = 0.0;
    float noise_lv = 0.0;
    float sampling_rate = 0.0;

    igl::viewer::Viewer viewer;

    viewer.callback_init = [&](igl::viewer::Viewer& viewer)
    {
        // Add an additional menu window
        viewer.ngui->addWindow(Eigen::Vector2i(900,10),"Task 1 panel");

        viewer.ngui->addGroup("Task 1");

        // Add Reset button
        viewer.ngui->addButton("Reset meshes for Task 1",[&](){

            igl::readOFF("../tutorial/CW1/bun000.off", V1, F1);
            igl::readOFF("../tutorial/CW1/bun045.off", V2, F2);
            show_meshes(viewer,V1,V2,F1,F2);

        });

        viewer.ngui->addGroup("Task 2");
        // Rotation about different axis
        viewer.ngui->addVariable("Rotation-X",a_x);
        viewer.ngui->addVariable("Rotation-Y",a_y);
        viewer.ngui->addVariable("Rotation-Z",a_z);

        viewer.ngui->addButton("Reset task2 meshes with angle",[&](){
            igl::readOFF("../tutorial/CW1/bun000.off", V1, F1);
            igl::readOFF("../tutorial/CW1/bun000.off", V2, F2);

            V2 = apply_rotation(V2,a_x,a_y,a_z);

            show_meshes(viewer,V1,V2,F1,F2);
        });

        // Noisy Model
        viewer.ngui->addGroup("Task 3");
        viewer.ngui->addVariable("Noise Level",noise_lv);
        viewer.ngui->addButton("Reset task3 meshes with noise",[&](){
            igl::readOFF("../tutorial/CW1/bun000.off", V1, F1);
            igl::readOFF("../tutorial/CW1/bun045.off", V2, F2);

            V2 = apply_noise(V2,noise_lv);

            show_meshes(viewer,V1,V2,F1,F2);
        });

        // Sampling
        viewer.ngui->addGroup("Task 4");
        viewer.ngui->addVariable("Sampling rate",sampling_rate);
        viewer.ngui->addButton("Reset task4 meshes with sampling",[&](){

            igl::readOFF("../tutorial/CW1/bun000.off", V1, F1);
            igl::readOFF("../tutorial/CW1/bun045.off", V2, F2);

            samples = uniform_sampling(V2,sampling_rate);

            show_meshes(viewer,V1,V2,F1,F2);

            viewer.data.add_points(samples,Eigen::RowVector3d(0,0,1));
        });
        viewer.ngui->addButton("Perform P2P ICP with sampling",[&](){
            performICP(V1,V2,200,true,sampling_rate);
            show_meshes(viewer,V1,V2,F1,F2);
            viewer.data.add_points(samples,Eigen::RowVector3d(0,0,1));
            std::cout<<"Resulting P2P error:"<<calculate_total_p2p_error(V1,V2)<<endl;
        });

        // Multiple scans
        viewer.ngui->addGroup("Task 5");
        viewer.ngui->addButton("Reset task5 of different scans",[&](){
            igl::readOFF("../tutorial/CW1/bun000.off", M1, Fm1);
            igl::readOFF("../tutorial/CW1/bun045.off", M2, Fm2);
            igl::readOFF("../tutorial/CW1/bun180.off", M3, Fm3);
            igl::readOFF("../tutorial/CW1/bun270.off", M4, Fm4);
            igl::readOFF("../tutorial/CW1/bun315.off", M5, Fm5);
            show_multiple_meshes(viewer);
        });

        viewer.ngui->addButton("Perform P2P ICP with multiple scans",[&](){

            Eigen::MatrixXd scans[5] = {M1,M2,M3,M4,M5};
            perform_ICP_with_multiple_scans(scans,5,0.02);

            M1 = scans[0];
            M2 = scans[1];
            M3 = scans[2];
            M4 = scans[3];
            M5 = scans[4];

            show_multiple_meshes(viewer);
        });

        viewer.ngui->addGroup("Task 6");
        viewer.ngui->addButton("Reset Meshes for task6",[&](){
            igl::readOFF("../tutorial/CW1/bun000.off", V1, F1);
            igl::readOFF("../tutorial/CW1/bun045.off", V2, F2);
            show_meshes(viewer,V1,V2,F1,F2);
        });
        viewer.ngui->addButton("Show Normal",[&](){
            viewer.data.clear();
            Eigen::MatrixXd Nv1 = estimateNormal(V1);
            show_meshes(viewer,V1,V2,F1,F2);
            viewer.data.add_edges(V1,V1+Nv1*.002,Eigen::RowVector3d(1,0,0));
        });
        viewer.ngui->addButton("Perform Point to Plane ICP",[&](){

            performPoint2PlaneICP(V1,V2,200);
            show_meshes(viewer,V1,V2,F1,F2);

            std::cout<<"Resulting P2P error:"<<calculate_total_p2p_error(V1,V2)<<endl;
        });

        viewer.ngui->addGroup("Point to point ICP Algorithm");
        // Add perform ICP for 1 iteration
        viewer.ngui->addButton("Perform P2P ICP for 1 iteration",[&](){
            performICP(V1,V2,1,false,0.0);
            show_meshes(viewer,V1,V2,F1,F2);
            std::cout<<"Resulting P2P error:"<<calculate_total_p2p_error(V1,V2)<<endl;
        });
        // Add perform ICP button
        viewer.ngui->addButton("Perform P2P ICP till converge",[&](){
            performICP(V1,V2,200,false,0.0);
            show_meshes(viewer,V1,V2,F1,F2);
            std::cout<<"Resulting P2P error:"<<calculate_total_p2p_error(V1,V2)<<endl;
        });


        // Generate menu
        viewer.screen->performLayout();

        return false;
    };

    show_meshes(viewer,V1,V2,F1,F2);

    viewer.launch();
}
