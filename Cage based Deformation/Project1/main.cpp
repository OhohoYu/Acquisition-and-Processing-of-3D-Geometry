#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <nanogui/formhelper.h>
#include <igl/edges.h>
#include <igl/writeOFF.h>
#include <nanogui/screen.h>
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <random>
#include <igl/PI.h>
#include <igl/jet.h>
#include "nanoflann.hpp"

using namespace Eigen;
using namespace std;

using namespace nanoflann;

typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, Dynamic, Dynamic> >  my_kd_tree_t;

Eigen::MatrixXd V,Control_V,CV1,CV2,Deform_V,D1,D2, w;
Eigen::MatrixXi F,Control_F,Deform_F;

char MeshName[] = "../tutorial/Project1/cow.off";
char MeshCutName[] = "../tutorial/Project1/cow_sliced.off";
char ControlMeshName[] = "../tutorial/Project1/cow_control.off";
char Deform1[] = "../tutorial/Project1/test.off";
char Deform2[] = "../tutorial/Project1/test2.off";
char Deform3[] = "../tutorial/Project1/test3.off";
char MyCage[] = "../tutorial/Project1/geoSphere_500t.off";

double Epsilon = 1.0e-20;
double Cage_Bias = 0.1;
const size_t num_results = 8;

double calDeterminant(MatrixXd u0, MatrixXd u1, MatrixXd u2){

    double det = u0(0)*(u1(1)*u2(2)-u2(1)*u1(2))
                 -u0(1)*(u1(0)*u2(2)-u2(0)*u1(2))
                  +u0(2)*(u1(0)*u2(1)-u2(0)*u1(1));

    return det;
}

MatrixXd calBarycentricCoordinates(int index,int total, MatrixXi control_face, double theta0,double theta1,double theta2,double l0,double l1,double l2){
    // clear the rest of weights to 0, only uses barycentric coordinates of three nodes
    MatrixXd weights = MatrixXd::Zero(total,1);

    //calculates barycentric coordinates as weights
    weights(control_face(index,0)) = sin(theta0) * l1 * l2;
    weights(control_face(index,1)) = sin(theta1) * l0 * l2;
    weights(control_face(index,2)) = sin(theta2) * l0 * l1;

    return weights;
}

// computes individual vertex weight given the control mesh
MatrixXd calVertexMeanValues(MatrixXd x, MatrixXd control_mesh, MatrixXi control_face){

    MatrixXd weights = MatrixXd::Zero(1,control_mesh.rows());

    // checks for vertices that is very close to the control mesh nodes
    for (int i = 0; i < control_mesh.rows() ; i++) {

        MatrixXd p = control_mesh.row(i);

        Vector3d d(x(0)-p(0),x(1)-p(1),x(2)-p(2));

        if(d.norm() <= Epsilon){
            weights(i) = 1.0;
            return weights;
        }
    }

    // if no nodes is too close to the control mesh
    // loop each face to get weight
    for (int j = 0; j < control_face.rows() ; j++) {
        double w0,w1,w2;

        MatrixXd p0 = control_mesh.row(control_face(j,0));
        MatrixXd p1 = control_mesh.row(control_face(j,1));
        MatrixXd p2 = control_mesh.row(control_face(j,2));

        Vector3d d0(x(0)-p0(0),x(1)-p0(1),x(2)-p0(2));
        Vector3d d1(x(0)-p1(0),x(1)-p1(1),x(2)-p1(2));
        Vector3d d2(x(0)-p2(0),x(1)-p2(1),x(2)-p2(2));

        // get unit vector of each edge in the triangles
        Vector3d u0 = d0.normalized();
        Vector3d u1 = d1.normalized();
        Vector3d u2 = d2.normalized();

        double l0 = (u1-u2).norm();
        double l1 = (u0-u2).norm();
        double l2 = (u0-u1).norm();

        double theta0 = 2 * asin(l0 * 0.5);
        double theta1 = 2 * asin(l1 * 0.5);
        double theta2 = 2 * asin(l2 * 0.5);

        // calculates the h parameters
        double h = (theta0+theta1+theta2)*0.5;

        if((M_PI - h) <= Epsilon){
            // if h is close to PI, then x lies on the triangle T, Hence barycentric coordinates are used
            return calBarycentricCoordinates(j,control_mesh.rows(),control_face,theta0,theta1,theta2,l0,l1,l2);
        }

        // 3d mean value vector within spherical triangle T
        double c0 = (2*sin(h)*sin(h-theta0))/(sin(theta1)*sin(theta2)) - 1.0;
        double c1 = (2*sin(h)*sin(h-theta1))/(sin(theta0)*sin(theta2)) - 1.0;
        double c2 = (2*sin(h)*sin(h-theta2))/(sin(theta0)*sin(theta1)) - 1.0;

        double det = calDeterminant(u0,u1,u2);

        double sign = (det<Epsilon)? -1 : 1;

        double s0 = sign * det * sqrt(1-c0*c0);
        double s1 = sign * det * sqrt(1-c1*c1);
        double s2 = sign * det * sqrt(1-c2*c2);

        // x lies outside triangle T but in the same plane as T, hence ignore
        if(s0<=Epsilon||s1<=Epsilon||s2<=Epsilon) {
            continue;
        }

        // other wise x lies in the control mesh interior
        // sum up the weights of different nodes
        weights(control_face(j,0)) += (theta0 - c1 * theta2 - c2*theta1) / (2 * d0.norm() * sin(theta2) * sqrt(1 - c1*c1));
        weights(control_face(j,1)) += (theta1 - c0 * theta2 - c2*theta0) / (2 * d1.norm() * sin(theta0) * sqrt(1 - c2*c2));
        weights(control_face(j,2)) += (theta2 - c1 * theta0 - c0*theta1) / (2 * d2.norm() * sin(theta1) * sqrt(1 - c0*c0));

    }

    return weights;
}

// Computes the weight map of the raw mesh to the control cage
MatrixXd calMeshMeanValues(MatrixXd M,MatrixXd control_mesh,MatrixXi control_face){
    MatrixXd weights = MatrixXd::Zero(M.rows(),control_mesh.rows());
    for (int i = 0; i < M.rows(); i++) {
        // computes the individual vertex weight
        weights.row(i) = calVertexMeanValues(M.row(i),control_mesh,control_face);
    }
    return weights;
}

// Given the weights and deformed control cage, computes the new deformed mesh
MatrixXd applyMVC2ControlMesh(MatrixXd W, MatrixXd control_mesh){
    MatrixXd P = MatrixXd::Zero(W.rows(),3);

    for (int i = 0; i < W.rows() ; i++) {
        MatrixXd row = MatrixXd::Zero(1,3);
        double weightSum = 0.0;
        for (int j = 0; j <W.cols() ; j++) {
            row += W(i,j)*control_mesh.row(j);
            weightSum += W(i,j);
        }
        //f(v) = sum(w*f)/sum(w), reconstruct the vertices using weighted
        //average of the control vertices coordinates
        P.row(i) = row/weightSum;
    }

    return P;
}

void loadCage(MatrixXd vertices,MatrixXi faces,MatrixXd& V1,MatrixXd& V2){
    MatrixXi Edges;
    // init the edges of the control mesh
    igl::edges(faces,Edges);
    V1 = MatrixXd(Edges.rows(),3);
    V2 = MatrixXd(Edges.rows(),3);

    for (int i = 0; i < Edges.rows(); i++) {
        V1.row(i) = vertices.row(Edges(i,0));
        V2.row(i) = vertices.row(Edges(i,1));
    }
}

// compute the max/min bounding box as color interpolation
MatrixXd computeColorMap(MatrixXd mesh){

    MatrixXd C = MatrixXd(mesh.rows(),3);

    // maximum and minimum coordinates of the mesh
    MatrixXd min = mesh.colwise().minCoeff();
    MatrixXd max = mesh.colwise().maxCoeff();

    double dx = max(0)-min(0);
    double dy = max(1)-min(1);
    double dz = max(2)-min(2);

    for (int i = 0; i < mesh.rows(); i++) {
        C(i,0) = (mesh(i,0)-min(0))/dx;
        C(i,1) = (mesh(i,1)-min(1))/dy;
        C(i,2) = (mesh(i,2)-min(2))/dz;
    }

    return C;
}

// compute the max/min bounding box as color interpolation scheme of a given mesh
MatrixXd interpolateColor(MatrixXd mesh, MatrixXd controlMesh, MatrixXd W){

    MatrixXd CMC = computeColorMap(controlMesh);

    MatrixXd C = MatrixXd(mesh.rows(),3);

    for (int i = 0; i < W.rows() ; i++) {
        MatrixXd row = MatrixXd::Zero(1,3);
        double weightSum = 0.0;
        for (int j = 0; j <W.cols() ; j++) {
            row += W(i,j) * CMC.row(j);
            weightSum += W(i,j);
        }

        C.row(i) = row / weightSum;
    }

    return C;

}

// compute the colour interpolation of sliced mesh
MatrixXd interpolateSlice(MatrixXd mesh, MatrixXd controlMesh, MatrixXi controlFaces){
    MatrixXd CMC = computeColorMap(controlMesh);

    MatrixXd C = MatrixXd(mesh.rows(),3);

    for (int i = 0; i <mesh.rows() ; i++) {
        // computes the weight w.r.t the cage
        MatrixXd weight = calVertexMeanValues(mesh.row(i),controlMesh,controlFaces);
        MatrixXd row = MatrixXd::Zero(1,3);

        double weightSum = 0.0;

        // reconstruct the colour coordinate using the cage colour coordinates
        for (int j = 0; j <weight.cols() ; j++) {
            row += weight(j) * CMC.row(j);
            weightSum += weight(j);
        }

        C.row(i) = row / weightSum;
    }

    return C;

}

void iterateSphereBBox(MatrixXd mesh, MatrixXd& control_mesh,bool jitter){

    // Build KD-tree for all vertices in P
    size_t nSamples = mesh.rows();

    Eigen::Matrix<double, Dynamic, Dynamic>  mat(nSamples, 3);

    for (size_t idx = 0; idx < nSamples; idx++)
        for (size_t d = 0; d < 3; d++)
            mat(idx, d) = mesh(idx, d);

    // Set the depth of the kd-three to be 10
    my_kd_tree_t   mat_index(mat, 10 /* max leaf */ );
    mat_index.index->buildIndex();



    for (int i = 0; i <control_mesh.rows() ; i++) {

        Vector3d control_point = control_mesh.row(i);

        std::vector<double> query_pt(3);
        // iterates dimension
        for (size_t d = 0; d < 3; d++)
            query_pt[d] = control_point(d);

        vector<size_t>   ret_indexes(num_results);
        vector<double> out_dists_sqr(num_results);

        nanoflann::KNNResultSet<double> resultSet(num_results);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
        mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        Vector3d mean_v = mesh.row(ret_indexes[0]);
        Vector3d diff = mean_v - control_point;
        Vector3d jittering = diff*(Eigen::MatrixXd::Random(3,3)- .5 * Eigen::MatrixXd::Ones(3,3));

        double length = diff.norm();
        if(length<=Cage_Bias) {
            continue;
        } else{
            double direct_scale = jitter ?  ((double) rand() / (RAND_MAX)) + 0.5 : 0.5 ;

            double jittering_scale = jitter ? ((double) rand() / (RAND_MAX)) - 0.25 * direct_scale : 0.0;

            control_mesh.row(i) += diff.normalized() * 0.5 * length + jittering.normalized() * length * jittering_scale;
        }


    }


}


// fit a spherical bounding box of the mesh by scaling and translation
void fitSphere(MatrixXd mesh, MatrixXd& control_sphere, double bias){

    MatrixXd m_min = mesh.colwise().minCoeff();
    MatrixXd m_max = mesh.colwise().maxCoeff();

    Vector3d vm_max(m_max(0)-m_min(0),m_max(1)-m_min(1),m_max(2)-m_min(2));

    // computes the centre of the mesh based on the bounding box
    Vector3d centre_m(m_min(0)+(m_max(0)-m_min(0))*0.5,
                      m_min(1)+(m_max(1)-m_min(1))*0.5,
                      m_min(2)+(m_max(2)-m_min(2))*0.5);

    // get the lenth of the bounding box conners of the mesh
    double m_R = vm_max.norm();

    MatrixXd s_min = control_sphere.colwise().minCoeff();
    MatrixXd s_max = control_sphere.colwise().maxCoeff();

    Vector3d vs_max(s_max(0)-s_min(0),s_max(1)-s_min(1),s_max(2)-s_min(2));

    // computes the centre of the control mesh based on the bounding box
    Vector3d centre_s(s_min(0)+(s_max(0)-s_min(0))*0.5,
                      s_min(1)+(s_max(1)-s_min(1))*0.5,
                      s_min(2)+(s_max(2)-s_min(2))*0.5);

    // computes the translation vector based on displacement of the centre of two meshes
    VectorXd T = centre_m-centre_s;

    // get the lenth of the bounding box conners of the control mesh
    double s_R = vs_max.norm();

    // including bias into the scaling factor
    double scale = m_R / s_R * bias;

    MatrixXd scaleMatrix(3,3);
    scaleMatrix << scale,0.0,0.0,
                    0.0,scale,0.0,
                    0.0,0.0,scale;

    control_sphere *= scaleMatrix;

    control_sphere.rowwise() += T.transpose();

}



int main(int argc, char *argv[])
{
    bool isControlCageShow = false;
    bool isDeformCage1Show = false;
    bool isDeformCage2Show = false;
    bool isDeformCage3Show = false;

    // Load a mesh in OFF format
    igl::readOFF(MeshName, V, F);
    igl::readOFF(ControlMeshName, Control_V,Control_F);
    loadCage(Control_V,Control_F,CV1,CV2);

    w = calMeshMeanValues(V,Control_V,Control_F);

    // Plot the mesh
    igl::viewer::Viewer viewer;

    // Add NanoGUI init callbacks
    viewer.callback_init = [&](igl::viewer::Viewer& viewer){
        // Add an additional menu window
        viewer.ngui->addWindow(Eigen::Vector2i(900,10),"Test");

        viewer.ngui->addGroup("Test");

        // Add Reset button
        viewer.ngui->addButton("Reset meshes",[&](){
            igl::readOFF(MeshName, V, F);
            viewer.data.clear();
            viewer.data.set_mesh(V,F);
            viewer.core.align_camera_center(V,F);
            isControlCageShow=false;
            isDeformCage1Show=false;
            isDeformCage2Show=false;
            isDeformCage3Show=false;

        });

        viewer.ngui->addVariable<bool>("Show Control Cage",[&](bool val) {
            isControlCageShow = val; // set

            if(isControlCageShow) {
                viewer.data.add_edges(CV1, CV2, RowVector3d(1, 0, 0));
            }
            else {
                viewer.data.clear();
                viewer.data.set_mesh(V,F);
                viewer.core.align_camera_center(V,F);
            }

        },[&]() {
            return isControlCageShow; // get
        });

        viewer.ngui->addGroup("Test Deform cage 1");
        viewer.ngui->addVariable<bool>("Show Deformed Cage 1",[&](bool val) {
            isDeformCage1Show = val; // set

            if(isDeformCage1Show) {
                igl::readOFF(Deform1,Deform_V,Deform_F);
                loadCage(Deform_V,Deform_F,D1,D2);
                viewer.data.add_edges(D1, D2, RowVector3d(0, 1, 0));
            }
            else {
                viewer.data.clear();
                viewer.data.set_mesh(V,F);
                viewer.core.align_camera_center(V,F);
            }

        },[&]() {
            return isDeformCage1Show; // get
        });

        viewer.ngui->addButton("Calculates mean value coordinates",[&](){
            igl::readOFF(Deform1,Deform_V,Deform_F);
            loadCage(Deform_V,Deform_F,D1,D2);
            V = applyMVC2ControlMesh(w,Deform_V);
            viewer.data.set_mesh(V,F);
            viewer.core.align_camera_center(V,F);
        });

        viewer.ngui->addGroup("Test Deform cage 2");
        viewer.ngui->addVariable<bool>("Show Deformed Cage 2",[&](bool val) {
            isDeformCage2Show = val; // set

            if(isDeformCage2Show) {
                igl::readOFF(Deform2,Deform_V,Deform_F);
                loadCage(Deform_V,Deform_F,D1,D2);
                viewer.data.add_edges(D1, D2, RowVector3d(0, 1, 0));
            }
            else {
                viewer.data.clear();
                viewer.data.set_mesh(V,F);
                viewer.core.align_camera_center(V,F);
            }

        },[&]() {
            return isDeformCage2Show; // get
        });

        viewer.ngui->addButton("Calculates mean value coordinates",[&](){
            igl::readOFF(Deform2,Deform_V,Deform_F);
            loadCage(Deform_V,Deform_F,D1,D2);
            V = applyMVC2ControlMesh(w,Deform_V);
            viewer.data.set_mesh(V,F);
            viewer.core.align_camera_center(V,F);
        });

        viewer.ngui->addGroup("Test Deform cage 3");
        viewer.ngui->addVariable<bool>("Show Deformed Cage 3",[&](bool val) {
            isDeformCage3Show = val; // set

            if(isDeformCage3Show) {
                igl::readOFF(Deform3,Deform_V,Deform_F);
                loadCage(Deform_V,Deform_F,D1,D2);
                viewer.data.add_edges(D1, D2, RowVector3d(0, 1, 0));
            }
            else {
                viewer.data.clear();
                viewer.data.set_mesh(V,F);
                viewer.core.align_camera_center(V,F);
            }

        },[&]() {
            return isDeformCage3Show; // get
        });

        viewer.ngui->addButton("Calculates mean value coordinates",[&](){
            igl::readOFF(Deform3,Deform_V,Deform_F);
            loadCage(Deform_V,Deform_F,D1,D2);
            V = applyMVC2ControlMesh(w,Deform_V);
            viewer.data.set_mesh(V,F);
            viewer.core.align_camera_center(V,F);
        });

        viewer.ngui->addButton("Hue Interpolation",[&](){
            igl::readOFF(MeshName, V, F);
            MatrixXd C = interpolateColor(V,Control_V,w);
            viewer.data.clear();
            viewer.data.set_mesh(V,F);
            viewer.data.set_colors(C);
            viewer.core.align_camera_center(V,F);

        });

        viewer.ngui->addButton("Hue Interpolation cuts",[&](){
            igl::readOFF(MeshCutName,V,F);
            MatrixXd C = interpolateSlice(V,Control_V,Control_F);

            viewer.data.clear();
            viewer.data.set_mesh(V,F);
            viewer.data.set_colors(C);
            viewer.core.align_camera_center(V,F);
        });

        viewer.ngui->addGroup("Build Cage");

        viewer.ngui->addButton("Show My Cage",[&](){
            igl::readOFF(MyCage,Deform_V,Deform_F);
            fitSphere(V,Deform_V,1.6);
            loadCage(Deform_V,Deform_F,D1,D2);
            viewer.data.add_edges(D1, D2, RowVector3d(0, 0, 1));
        });

        viewer.ngui->addButton("Iterate Cage",[&](){
            iterateSphereBBox(V, Deform_V,true);

            viewer.data.clear();
            viewer.data.set_mesh(V,F);
            loadCage(Deform_V,Deform_F,D1,D2);
            viewer.data.add_edges(D1, D2, RowVector3d(0, 0, 1));
        });

        viewer.ngui->addButton("Export Cage",[&](){
            igl::writeOFF("MyCage.off",Deform_V,Deform_F);
        });



        viewer.screen->performLayout();

        return false;
    };


    viewer.data.set_mesh(V, F);
    viewer.launch();

}

