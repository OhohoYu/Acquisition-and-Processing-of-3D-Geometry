#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/adjacency_list.h>
#include <igl/vertex_triangle_adjacency.h>
#include <nanogui/screen.h>
#include <nanogui/formhelper.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <cstdlib>
#include <iostream>
#include <igl/jet.h>
#include <igl/PI.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <SymEigsSolver.h>
#include <MatOp/SparseSymMatProd.h>



Eigen::MatrixXd V;
Eigen::MatrixXi F;

typedef Eigen::Triplet<double> T;

using namespace Spectra;
using namespace Eigen;
using namespace std;

Eigen::MatrixXd smoothWithExplictScheme(Eigen::MatrixXd P,Eigen::SparseMatrix<double> L,double lamda){
    Eigen::MatrixXd Q(P.rows(),P.cols());
    Q = ((MatrixXd::Identity(P.rows(),P.rows()))+ lamda * MatrixXd(L))*P;
    return Q;
}

Eigen::MatrixXd smoothWithImplictScheme(Eigen::MatrixXd P,Eigen::SparseMatrix<double> L,double lamda){
    Eigen::MatrixXd Q(P.rows(),P.cols());
    Eigen::MatrixXd M(P.rows(),P.cols());
    M = ((MatrixXd::Identity(P.rows(),P.rows()))- lamda * MatrixXd(L));
    Q = M.inverse()*P;
    return Q;
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


Eigen::MatrixXd computeGuassianCurvature(Eigen::MatrixXd P,Eigen::MatrixXi Fp){

    Eigen::MatrixXd K(P.rows(),1);

    std::vector<std::vector<int> > VF,VFi;
    Eigen::MatrixXd mass(P.rows(),1);
    igl::vertex_triangle_adjacency(P,Fp,VF,VFi);


    int i=0;
    for(auto& row:VF){
        double theta = 2*igl::PI;
        double A = 0.0;

        for(auto& col:row){
            MatrixXi f = Fp.row(col);
            int Ib = (f(0)==i)?f(1):f(0);
            int Ic = (f(2)==i)?f(1):f(2);

            Vector3d v1(P(Ib,0)-P(i,0),P(Ib,1)-P(i,1),P(Ib,2)-P(i,2));
            Vector3d v2(P(Ic,0)-P(i,0),P(Ic,1)-P(i,1),P(Ic,2)-P(i,2));

            // Angle = acos(a.b/|a||b|)
            double angle = acos(v1.dot(v2)/(v1.norm()*v2.norm()));
            // Area = |a||b|sin(c)/2
            double area = v1.norm()*v2.norm()*sin(angle)*0.5;

            A += area/3.0;

            theta -= angle;
        }
        // normlize by area
        K(i) = theta/A;
        i++;
    }

    return K;

}


// this function returns the 2 vectices that shares same edges, by intersecting their neigborhood
vector<int> intersection(vector<int> &v1, vector<int> &v2){

    vector<int> v3;

    sort(v1.begin(), v1.end());
    sort(v2.begin(), v2.end());

    set_intersection(v1.begin(),v1.end(),v2.begin(),v2.end(),back_inserter(v3));

    return v3;
}


Eigen::SparseMatrix<double> computeNonUniformLaplace(Eigen::MatrixXd P,Eigen::MatrixXi Fp){
    size_t nSamples = P.rows();

    std::vector<T> tempM;
    std::vector<T> tempS;

    std::vector<std::vector<int> > V2V;
    igl::adjacency_list(Fp,V2V);

    int i = 0;

    for(auto& row:V2V){
        double A = 0.0;
        double theta = 0.0;

       for(auto& col:row){
           std::vector<int> target = V2V[col];
           // find the vertices that shares edge OA
           vector<int> neighbor = intersection(row,target);

           int B = neighbor[0];
           int C = neighbor[1];

           // Angle = acos(a.b/|a||b|)
           Vector3d BO(P(i,0)-P(B,0),P(i,1)-P(B,1),P(i,2)-P(B,2));
           // Area = |a||b|sin(c)/2
           Vector3d BA(P(col,0)-P(B,0),P(col,1)-P(B,1),P(col,2)-P(B,2));

           double alpha = acos(BO.dot(BA)/(BO.norm()*BA.norm()));
           double areaB = BO.norm()*BA.norm()*sin(alpha)*0.5;

           A += areaB/3.0;

           Vector3d CO(P(i,0)-P(C,0),P(i,1)-P(C,1),P(i,2)-P(C,2));
           Vector3d CA(P(col,0)-P(C,0),P(col,1)-P(C,1),P(col,2)-P(C,2));

           double beta = acos(CO.dot(CA)/(CO.norm()*CA.norm()));
           double areaC = CO.norm()*CA.norm()*sin(beta)*0.5;

           A += areaC/3.0;

           double scalar = cos(alpha)/sin(alpha) + cos(beta)/sin(beta);

           tempS.push_back(T(i,col,scalar));
           theta += scalar;

       }

        tempS.push_back(T(i,i,-theta));
       // Area A is added twice, so the normalize term for M^-1 is 1/2A
        tempM.push_back(T(i,i,1.0/A));

       i++;
    }

    Eigen::SparseMatrix<double> TestM(nSamples,nSamples);
    TestM.setFromTriplets(tempM.begin(), tempM.end());
    Eigen::SparseMatrix<double> TestS(nSamples,nSamples);
    TestS.setFromTriplets(tempS.begin(), tempS.end());

    return TestM*TestS;
}

Eigen::SparseMatrix<double> computeUniformLaplace(Eigen::MatrixXd P,Eigen::MatrixXi Fp){
    size_t nSamples = P.rows();

    std::vector<std::vector<int> > V2V;
    igl::adjacency_list(Fp,V2V);

    std::vector<T> tempL;

    int i = 0;
    for(auto& row:V2V){
        tempL.push_back(T(i,i,-1.0));
        double n = 1.0/(double)row.size();
        for(auto& col:row){
            tempL.push_back(T(i,col,n));
        }
        i++;
    }
    Eigen::SparseMatrix<double> testL(nSamples,nSamples);
    testL.setFromTriplets(tempL.begin(), tempL.end());

    return testL;
};


Eigen::MatrixXd  reconstructWithSpectral(Eigen::MatrixXd P,Eigen::MatrixXi Fp,int k) {
    Eigen::MatrixXd Q = MatrixXd::Zero(P.rows(),3);
    https://doodle.com/poll/kyhuipb5qwzkatvf    // compute the M^1/2 and M^-1/2 diagnal matrix and S the symmetric matrix
    //================================================================================

    size_t nSamples = P.rows();
    SparseMatrix<double> L(nSamples,nSamples);

    std::vector<T> tempM_half,tempM_inv_half;
    std::vector<T> tempS;

    std::vector<std::vector<int> > V2V;
    igl::adjacency_list(Fp,V2V);

    int i = 0;

    for(auto& row:V2V){
        double A = 0.0;
        double theta = 0.0;

        for(auto& col:row){
            std::vector<int> target = V2V[col];
            vector<int> neighbor = intersection(row,target);

            int B = neighbor[0];
            int C = neighbor[1];

            Vector3d BO(P(i,0)-P(B,0),P(i,1)-P(B,1),P(i,2)-P(B,2));
            Vector3d BA(P(col,0)-P(B,0),P(col,1)-P(B,1),P(col,2)-P(B,2));

            double alpha = acos(BO.dot(BA)/(BO.norm()*BA.norm()));
            double areaB = BO.norm()*BA.norm()*sin(alpha)*0.5;

            A += areaB/3.0;

            Vector3d CO(P(i,0)-P(C,0),P(i,1)-P(C,1),P(i,2)-P(C,2));
            Vector3d CA(P(col,0)-P(C,0),P(col,1)-P(C,1),P(col,2)-P(C,2));

            double beta = acos(CO.dot(CA)/(CO.norm()*CA.norm()));
            double areaC = CO.norm()*CA.norm()*sin(beta)*0.5;

            A += areaC/3.0;

            double scalar = cos(alpha)/sin(alpha) + cos(beta)/sin(beta);

            tempS.push_back(T(i,col,scalar));
            theta += scalar;

        }

        tempS.push_back(T(i,i,-theta));

        tempM_inv_half.push_back(T(i,i,1.0/sqrt(A)));
        tempM_half.push_back(T(i,i,sqrt(A)));

        i++;
    }

    Eigen::SparseMatrix<double> M_inv_half(nSamples,nSamples);
    M_inv_half.setFromTriplets(tempM_inv_half.begin(), tempM_inv_half.end());

    Eigen::SparseMatrix<double> M_half(nSamples,nSamples);
    M_half.setFromTriplets(tempM_half.begin(), tempM_half.end());

    Eigen::SparseMatrix<double> S(nSamples,nSamples);
    S.setFromTriplets(tempS.begin(), tempS.end());

    // Sym = M^-1/2 * C * M^-1/2
    Eigen::SparseMatrix<double> Sym = M_inv_half*S*M_inv_half;
    // Y = M^1/2 * X
    Eigen::MatrixXd Y = M_half*P;

    //==============================================================================

    SparseSymMatProd<double> op(Sym);
    // find the smallest magnitude eigenvector
    SymEigsSolver< double, SMALLEST_MAGN , SparseSymMatProd<double> > eigs(&op, k, 2*k+1);

    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    Eigen::VectorXd evalues;
    Eigen::MatrixXd evecs;
    if(eigs.info() == SUCCESSFUL) {
        evalues = eigs.eigenvalues();
        evecs = eigs.eigenvectors();
    }else{
        std::cout<<"Eigen solving Error"<<endl;
    }
    std::cout<<"Eigen values /n"<<endl;
    std::cout<<evalues<<endl;
    std::cout<<"Eigen vectors /n"<<endl;
    std::cout<<evecs<<endl;

    Eigen::MatrixXd tempX = MatrixXd::Zero(P.rows(),1);
    Eigen::MatrixXd tempY = MatrixXd::Zero(P.rows(),1);
    Eigen::MatrixXd tempZ = MatrixXd::Zero(P.rows(),1);

    // project to k-eigen subspace
    for(int j=0; j < k; j++){
        tempX +=  (Y.col(0).transpose()*evecs.col(j))*evecs.col(j);
        tempY +=  (Y.col(1).transpose()*evecs.col(j))*evecs.col(j);
        tempZ +=  (Y.col(2).transpose()*evecs.col(j))*evecs.col(j);
    }
    // X = M^-1/2 * Y  convert Y to X
    Q<<tempX,tempY,tempZ;
    Q = M_inv_half*Q;

    return Q;
}


int main(int argc, char *argv[])
{
    char MeshName[]="../tutorial/CW2/screwdriver.off";
  // Load a mesh in OFF format
  igl::readOFF(MeshName, V, F);

  igl::viewer::Viewer viewer;

  double lamda_explict = 0.1;
  double lamda_implict = 0.1;
  int    k =5;
  double noise_lvl = 0.01;

  viewer.callback_init = [&](igl::viewer::Viewer& viewer){
      // Add an additional menu window
      viewer.ngui->addWindow(Eigen::Vector2i(900,10),"Task 1 panel");

      viewer.ngui->addGroup("Uniform Laplace");

      // Add Reset button
      viewer.ngui->addButton("Reset meshes",[&](){

          igl::readOFF(MeshName, V, F);
          viewer.data.clear();
          viewer.data.set_mesh(V, F);
          viewer.core.align_camera_center(V,F);
      });

      viewer.ngui->addButton("Show Mean curvatures",[&](){
          Eigen::MatrixXd H;
          Eigen::MatrixXd C;
          Eigen::SparseMatrix<double> L;
          L = computeUniformLaplace(V,F);
          H = (L*V).rowwise().norm()*0.5;
          igl::jet(H,true,C);
          viewer.data.set_colors(C);
      });

      viewer.ngui->addButton("Show Gaussian curvatures",[&](){
          Eigen::MatrixXd K;
          Eigen::MatrixXd C;
          K = computeGuassianCurvature(V,F);
          igl::jet(K,true,C);
          viewer.data.set_colors(C);
      });

      viewer.ngui->addGroup("Non-uniform Laplace");

      viewer.ngui->addButton("Show Mean curvatures",[&](){
          Eigen::MatrixXd H;
          Eigen::MatrixXd C;
          Eigen::SparseMatrix<double> L;
          L = computeNonUniformLaplace(V,F);
          H = (L*V).rowwise().norm()*0.5;
          igl::jet(H,true,C);
          viewer.data.set_colors(C);
      });

      viewer.ngui->addGroup("Spectral Analysis");
      viewer.ngui->addVariable("k min eigs",k);
      viewer.ngui->addButton("Reconstruct",[&](){
          Eigen::MatrixXd Q;
          Q = reconstructWithSpectral(V,F,k);
          viewer.data.clear();
          viewer.data.set_mesh(Q,F);
          viewer.core.align_camera_center(Q,F);
          V = Q;
      });

      viewer.ngui->addGroup("Explict Smoothing");
      viewer.ngui->addVariable("Lamda",lamda_explict);

      viewer.ngui->addButton("Smooth for 1 iteration",[&](){
          Eigen::MatrixXd Q;
          Eigen::SparseMatrix<double> L;
          L = computeUniformLaplace(V,F);
          Q = smoothWithExplictScheme(V,L,lamda_explict);
          viewer.data.clear();
          viewer.data.set_mesh(Q,F);
          viewer.core.align_camera_center(Q,F);
          V = Q;
      });

      viewer.ngui->addGroup("Implicit Smoothing");
      viewer.ngui->addVariable("Lamda",lamda_implict);
      viewer.ngui->addButton("Smooth for 1 iteration",[&](){
          Eigen::MatrixXd Q;
          Eigen::SparseMatrix<double> L;
          L = computeUniformLaplace(V,F);
          Q = smoothWithImplictScheme(V,L,lamda_implict);
          viewer.data.clear();
          viewer.data.set_mesh(Q,F);
          viewer.core.align_camera_center(Q,F);
          V = Q;
      });

      viewer.ngui->addGroup("Test Noise");
      viewer.ngui->addVariable("Noise level",noise_lvl);
      viewer.ngui->addButton("Add Noise",[&](){
          Eigen::MatrixXd Q;
          Q = apply_noise(V,noise_lvl);
          viewer.data.clear();
          viewer.data.set_mesh(Q,F);
          viewer.core.align_camera_center(Q,F);
          V = Q;
      });

      viewer.screen->performLayout();

      return false;
  };
  viewer.data.set_mesh(V, F);
  viewer.launch();
}
