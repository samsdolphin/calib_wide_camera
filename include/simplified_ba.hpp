#ifndef BA_KEEP_ABLATION
#define BA_KEEP_ABLATION

#include "tools.hpp"
#include <thread>
#include <Eigen/Eigenvalues>
// #include "preintegration2.hpp"
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <opencv2/opencv.hpp>

#define WIN_SIZE 10
#define GAP 5
#define AVG_THR
// #define KEEP
#define USE_NEW_PCD
// #define HESSIAN_USE_CURVE

// const int jac_leng = DVEL * WIN_SIZE;
// const int imu_leng = DIM * WIN_SIZE;

const double one_three = (1.0 / 3.0);

int layer_limit = 3; // reject voxel if this->layer == layer_limit
// int layer_size[4] = {30, 25, 20, 15};
float eigen_value_array[4] = {1.0/20, 1.0/15, 1.0/10, 1.0/5}; // structured environment
int MIN_PT = 5;

int life_span = 1000;
int thd_num = 16;

class VOX_HESS
{
public:
  vector<const VOX_FACTOR*> sig_vecs;
  vector<const vector<VOX_FACTOR>*> plvec_voxels;
  vector<PLV(3)> origin_points;
  int win_size;

  VOX_HESS(int _win_size = WIN_SIZE): win_size(_win_size){origin_points.resize(win_size);}

  ~VOX_HESS()
  {
    vector<const VOX_FACTOR*>().swap(sig_vecs);
    vector<const vector<VOX_FACTOR>*>().swap(plvec_voxels);
  }

  void get_center(const PLV(3)& vec_orig, PLV(3)& origin_points_)
  {
    size_t pt_size = vec_orig.size();
    int filter = 4;
    // if(pt_size < filter)
    {
      for(size_t i = 0; i < pt_size; i++)
        origin_points_.emplace_back(vec_orig[i]);
      return;
    }

    Eigen::Vector3d center;
		double part = 1.0 * pt_size / filter;

		for(int i = 0; i < filter; i++)
		{
			size_t np = part * i;
			size_t nn = part * (i + 1);
			center.setZero();
			for(size_t j = np; j < nn; j++)
				center += vec_orig[j];

			center = center / (nn - np);
			origin_points_.emplace_back(center);
		}
  }

  void push_voxel(const vector<VOX_FACTOR>* sig_orig, const VOX_FACTOR* fix,
                  const vector<PLV(3)>* vec_orig, double feat_eigen, int layer)
  {
    int process_size = 0;
    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        process_size++;

    #ifdef USE_NEW_PCD
    if(process_size < 1) return;

    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        get_center((*vec_orig)[i], origin_points[i]);
    #endif
    
    if(process_size < 2) return;

    // for(int i = 0; i < win_size-1; i++)
    //   if((*sig_orig)[i].N != 0)
    //     for(int j = i+1; j < win_size; i++)
    //       if((*sig_orig)[j].N != 0)
    //         if(j!=i+1 && j!=i+GAP && j!=i+GAP*GAP)
    //           return;
    
    plvec_voxels.push_back(sig_orig);
    sig_vecs.push_back(fix);
  }

  Eigen::Matrix<double, 6, 1> lam_f(Eigen::Vector3d *u, int m, int n)
  {
    Eigen::Matrix<double, 6, 1> jac;
    jac[0] = u[m][0] * u[n][0];
    jac[1] = u[m][0] * u[n][1] + u[m][1] * u[n][0];
    jac[2] = u[m][0] * u[n][2] + u[m][2] * u[n][0];
    jac[3] = u[m][1] * u[n][1];
    jac[4] = u[m][1] * u[n][2] + u[m][2] * u[n][1];
    jac[5] = u[m][2] * u[n][2];
    return jac;
  }

  void acc_evaluate2(const vector<IMUST>& xs, int head, int end,
                     Eigen::MatrixXd& Hess, Eigen::VectorXd& JacT, double& residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a = head; a < end; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];

      VOX_FACTOR sig = *sig_vecs[a];
      for(int i = 0; i < win_size; i++)
      if(sig_orig[i].N != 0)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }
      
      const Eigen::Vector3d& vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d& lmbd = saes.eigenvalues();
      const Eigen::Matrix3d& U = saes.eigenvectors();
      int NN = sig.N;
      
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d& uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i = 0; i < 3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i = 0; i < win_size; i++)
        if(sig_orig[i].N != 0)
        {
          Eigen::Matrix3d Pi = sig_orig[i].P;
          Eigen::Vector3d vi = sig_orig[i].v;
          Eigen::Matrix3d Ri = xs[i].R;
          double ni = sig_orig[i].N;

          Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
          Eigen::Vector3d RiTuk = Ri.transpose() * uk;
          Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();
          
          Eigen::Vector3d ti_v = xs[i].p - vBar;
          double ukTti_v = uk.dot(ti_v);

          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          // for(int j = 0; j < win_size; j++)
          //   if(sig_orig[j].N != 0)
          //     if(j%5 == 0 || j%25 == 0)
          //       JacT.block<6, 1>(6*i, 0) += jjt;
          JacT.block<6, 1>(6*i, 0) += jjt;

          const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          Hb.block<3, 3>(0, 0) +=
            2.0/NN*(combo1-RiTukhat*Pi)*RiTukhat - 2.0/NN/NN*viRiTuk[i]*viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
          Hb.block<3, 3>(0, 3) += HRt;
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

          Hess.block<6, 6>(6*i, 6*i) += Hb;
        }
      
      for(int i = 0; i < win_size-1; i++)
        if(sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          for(int j = i+1; j < win_size; j++)
            if(sig_orig[j].N != 0)
            {
              // if(j!=i+1 && j!=i+GAP && j!=i+GAP*GAP) continue;
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

              Hess.block<6, 6>(6*i, 6*j) += Hb;
            }
        }
      
      residual += lmbd[kk];
    }

    for(int i = 1; i < win_size; i++)
      for(int j = 0; j < i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  void evaluate_only_residual(const vector<IMUST>& xs, double& residual)
  {
    residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig = *sig_vecs[a];

      for(int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += lmbd[kk];
    }
  }

  std::vector<double> evaluate_residual(const vector<IMUST>& xs)
  {
    std::vector<double> residuals;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig = *sig_vecs[a];

      for(int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residuals.push_back(lmbd[kk]);
      // residuals.push_back(sigmoid_w(lmbd[kk]));
    }
    assert(residuals.size() == gps_size);
    return residuals;
  }

  void remove_residual(const vector<IMUST>& xs, double threshold, double reject_num)
  {
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int rej_cnt = 0;
    size_t i = 0;
    for(; i < plvec_voxels.size();)
    {
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[i];
      VOX_FACTOR sig = *sig_vecs[i];

      for(int j = 0; j < win_size; j++)
      {
        sig_tran[j].transform(sig_orig[j], xs[j]);
        sig += sig_tran[j];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      if(lmbd[kk] >= threshold)
      {
        plvec_voxels.erase(plvec_voxels.begin()+i);
        sig_vecs.erase(sig_vecs.begin()+i);
        rej_cnt++;
        // std::cout<<"reject "<<rej_cnt<<std::endl;
        continue;
      }
      i++;
      if(rej_cnt == reject_num) break;
    }
  }
};

enum OCTO_STATE {UNKNOWN, MID_NODE, PLANE};
class OCTO_TREE_NODE
{
public:
  OCTO_STATE octo_state;
  int push_state;
  int layer, win_size;
  vector<PLV(3)> vec_orig, vec_tran;
  vector<VOX_FACTOR> sig_orig, sig_tran;
  VOX_FACTOR fix_point;

  OCTO_TREE_NODE* leaves[8];
  float voxel_center[3];
  float quater_length;
  float eigen_thr;

  Eigen::Vector3d center, direct, value_vector; // temporal
  double eigen_ratio;
  
  #ifdef KEEP
  ros::NodeHandle nh;
  ros::Publisher pub_residual = nh.advertise<sensor_msgs::PointCloud2>("/residual", 100);
  #endif

  OCTO_TREE_NODE(int _win_size = WIN_SIZE, float _eigen_thr = 1.0/10):
    win_size(_win_size), eigen_thr(_eigen_thr)
  {
    octo_state = UNKNOWN; push_state = 0;
    vec_orig.resize(win_size); vec_tran.resize(win_size);
    sig_orig.resize(win_size); sig_tran.resize(win_size);
    for(int i = 0; i < 8; i++) leaves[i] = nullptr;
  }

  virtual ~OCTO_TREE_NODE()
  {
    vec_orig.clear(); vec_tran.clear();
    sig_orig.clear(); sig_tran.clear();
    vector<PLV(3)>().swap(vec_orig);
    vector<PLV(3)>().swap(vec_tran);
    vector<VOX_FACTOR>().swap(sig_orig);
    vector<VOX_FACTOR>().swap(sig_tran);
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  bool judge_eigen(int win_count)
  {
    VOX_FACTOR covMat = fix_point;
    for(int i = 0; i < win_count; i++)
      covMat += sig_tran[i];
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    center = covMat.v / covMat.N;
    direct = saes.eigenvectors().col(0);

    eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // [0] is the smallest
    // eigen_ratio = saes.eigenvalues()[0];

    // return eigen_ratio < eigen_value_array[layer];
    return eigen_ratio < eigen_thr;
  }

  void cut_func(PLV(3)& pvec_orig, PLV(3)& pvec_tran, int ci)
  {
    uint a_size = pvec_tran.size();
    for(uint j = 0; j < a_size; j++)
    {
      int xyz[3] = {0, 0, 0};
      for(uint k = 0; k < 3; k++)
        if(pvec_tran[j][k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OCTO_TREE_NODE(win_size, eigen_thr);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
        leaves[leafnum]->layer = layer + 1;
      }

      #ifndef KEEP
      if(leaves[leafnum]->octo_state!=PLANE && leaves[leafnum]->layer!=layer_limit)
      #endif
      {
        leaves[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
        leaves[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);
      }
      
      if(leaves[leafnum]->octo_state != MID_NODE)
      {
        leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]);
        leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
      }
    }

    PLV(3)().swap(pvec_orig);
    PLV(3)().swap(pvec_tran);
  }

  void recut(int win_count)
  {
    if(octo_state != MID_NODE) // 空voxel加入当前帧点云或者voxel之前是平面
    {
      int point_size = fix_point.N;
      for(int i = 0; i < win_count; i++)
        point_size += sig_orig[i].N;

      if(point_size <= MIN_PT)
      {
        push_state = 0;
        return;
      }

      if(judge_eigen(win_count))
      {
        // if(octo_state==0 && point_size>layer_size[layer])
        if(octo_state == UNKNOWN) // 当前帧点云是平面
        {
          #ifndef KEEP
          #ifndef USE_NEW_PCD
          vector<PLV(3)>().swap(vec_orig);
          #endif
          vector<PLV(3)>().swap(vec_tran);
          #endif
          octo_state = PLANE;
        }
        push_state = 1;
      }
      else if(octo_state == PLANE) // 当前帧点云进来后不再是平面，则剔除当前帧点云
      {
        point_size = point_size - (int)sig_orig[win_count-1].N;
        sig_orig[win_count-1].clear();
        sig_tran[win_count-1].clear();
        push_state = 0;
      }
      else // voxel之前不是平面，现在还不是平面
      {
        octo_state = MID_NODE;
        push_state = 0;
      }

      if(push_state==1 || octo_state==PLANE)
      {
        assert(fix_point.N==0);
        point_size -= fix_point.N;
        if(point_size <= MIN_PT) push_state = 0; 
        return;
      }

      if(layer == layer_limit)
      {
        #ifndef KEEP
        #ifndef USE_NEW_PCD
        vector<PLV(3)>().swap(vec_orig);
        #endif
        vector<PLV(3)>().swap(vec_tran);
        #endif
        octo_state = UNKNOWN;
        return;
      }
      else
      {
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        for(int i = 0; i < win_count; i++)
          cut_func(vec_orig[i], vec_tran[i], i);
      }
    }
    else // 之前voxel就不是平面，直接切割当前帧点云
      cut_func(vec_orig[win_count-1], vec_tran[win_count-1], win_count-1);
    
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut(win_count);
  }

  void tras_opt(VOX_HESS& vox_opt, int win_count)
  {
    if(octo_state != MID_NODE)
    {
      int points_size = 0;
      for(int i = 0; i < win_count; i++)
        points_size += sig_orig[i].N;
      
      if(points_size < MIN_PT) return;

      if(push_state == 1)
        vox_opt.push_voxel(&sig_orig, &fix_point, &vec_orig, eigen_ratio, layer);
    }
    else
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt, win_count);
  }

  void tras_display(int win_count, int layer = 0)
  {
    // float ref = 255.0*rand()/(RAND_MAX + 1.0f);
    // std::cout<<"ref "<<ref<<std::endl;
    pcl::PointXYZRGB ap;
    // ap.intensity = ref;

    if(octo_state == PLANE)
    {
      // if(push_state != 1) return;
      int total = 0;
      for(int i = 0; i < win_count; i++)
        total += vec_tran[i].size();

      std::vector<unsigned int> colors;
			colors.push_back(static_cast<unsigned int>(rand() % 256));
			colors.push_back(static_cast<unsigned int>(rand() % 256));
			colors.push_back(static_cast<unsigned int>(rand() % 256));
      pcl::PointCloud<pcl::PointXYZRGB> color_cloud;

      for(int i = 0; i < win_count; i++)
        for(size_t j = 0; j < vec_tran[i].size(); j++)
        {
          Eigen::Vector3d& pvec = vec_tran[i][j];
          ap.x = pvec.x();
          ap.y = pvec.y();
          ap.z = pvec.z();
          ap.b = colors[0];
          ap.g = colors[1];
          ap.r = colors[2];
          // ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
          // ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
          // ap.normal_z = sqrt(value_vector[0]);
          // ap.curvature = total;
          // std::cout<<"total "<<total<<std::endl;
          color_cloud.push_back(ap);
        }
      #ifdef KEEP
      sensor_msgs::PointCloud2 dbg_msg;
      pcl::toROSMsg(color_cloud, dbg_msg);
      dbg_msg.header.frame_id = "/camera_init";
      pub_residual.publish(dbg_msg);
      #endif
      // exit(0);
    }
    else
    {
      layer++;
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(win_count, layer);
    }
  }
};

class OCTO_TREE_ROOT: public OCTO_TREE_NODE
{
public:
  bool is2opt;
  int life;
  vector<int> each_num;

  OCTO_TREE_ROOT(int _winsize, float _eigen_thr): OCTO_TREE_NODE(_winsize, _eigen_thr)
  {
    is2opt = true;
    life = life_span;
    each_num.resize(win_size);
    for(int i = 0; i < win_size; i++)
      each_num[i] = 0;
  }

  virtual ~OCTO_TREE_ROOT(){}
};

double vel_coef = 0.1;
double imu_coef = 0.1;

class VOX_OPTIMIZER
{
public:
  int win_size, jac_leng, imu_leng;
  VOX_OPTIMIZER(int _win_size = WIN_SIZE): win_size(_win_size)
  {
    jac_leng = DVEL * win_size;
    imu_leng = DIM * win_size;
  }

  double divide_thread(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab,
                       Eigen::MatrixXd& Hess,Eigen::VectorXd& JacT)
  {
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);

    for(int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    for(int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1),
                               ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    // Eigen::Matrix<double, DVEL, 1> rr;
    // Eigen::Matrix<double, DVEL, DVEL> joca, jocb;

    // for(int i=1; i<WIN_SIZE; i++)
    // {
    //   joca.setZero(); jocb.setZero();

    //   Eigen::Matrix3d &Ra = x_stats[i-1].R;
    //   Eigen::Vector3d &pa = x_stats[i-1].p;
    //   Eigen::Matrix3d &Rb = x_stats[i].R;
    //   Eigen::Vector3d &pb = x_stats[i].p;
    //   Eigen::Matrix3d &Rab = x_ab[i].R;
    //   Eigen::Vector3d &pab = x_ab[i].p;

    //   Eigen::Matrix3d res_r = Rab.transpose() * Ra.transpose() * Rb;
    //   Eigen::Vector3d res_p = pb - pa;
    //   rr.setZero();
    //   rr.block<3, 1>(0, 0) = Log(res_r);
    //   rr.block<3, 1>(3, 0) = Ra.transpose() * res_p - pab;

    //   residual += rr.squaredNorm();

    //   Eigen::Matrix3d JR_inv = jr_inv(res_r);
    //   joca.block<3, 3>(0, 0) = -JR_inv * Rb.transpose() * Ra;
    //   jocb.block<3, 3>(0, 0) =  JR_inv;
    //   joca.block<3, 3>(3, 0) =  hat(Ra.transpose() * res_p);
    //   joca.block<3, 3>(3, 3) = -Ra.transpose();
    //   jocb.block<3, 3>(3, 3) =  Ra.transpose();

    //   Eigen::Matrix<double, 6, 12> joc;
    //   joc.block<6, 6>(0, 0) = joca;
    //   joc.block<6, 6>(0, 6) = jocb;

    //   Hess.block<12, 12>((i-1)*6, (i-1)*6) += joc.transpose() * joc;
    //   JacT.block<12, 1>((i-1)*6, 0) += joc.transpose() * rr;
    // }

    // rr.setZero(); joca.setIdentity();
    // Eigen::Matrix3d res_r = x_ab[0].R.transpose() * x_stats[0].R;
    // rr.block<3, 1>(0, 0) = Log(res_r);
    // rr.block<3, 1>(3, 0) = x_stats[0].p - x_ab[0].p;
    // joca.block<3, 3>(0, 0) = jr_inv(res_r);
    // Hess.block<DVEL, DVEL>(0, 0) += joca.transpose() * joca;
    // JacT.block<DVEL, 1>(0, 0) += joca.transpose() * rr;
    // residual += rr.squaredNorm();

    // Hess *= vel_coef;
    // JacT *= vel_coef;
    // residual *= (vel_coef * 0.5);

    for(int i=0; i<tthd_num; i++)
    {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }
    #ifdef AVG_THR
    return residual/g_size;
    #else
    return residual;
    #endif
  }

  double only_residual(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab, bool is_avg = false)
  {
    double residual1 = 0, residual2 = 0;

    // Eigen::Matrix<double, DVEL, 1> rr;
    // for(int i=1; i<WIN_SIZE; i++)
    // {
    //   Eigen::Matrix3d &Ra = x_stats[i-1].R;
    //   Eigen::Vector3d &pa = x_stats[i-1].p;
    //   Eigen::Matrix3d &Rb = x_stats[i].R;
    //   Eigen::Vector3d &pb = x_stats[i].p;
    //   Eigen::Matrix3d &Rab = x_ab[i].R;
    //   Eigen::Vector3d &pab = x_ab[i].p;

    //   Eigen::Matrix3d res_r = Rab.transpose() * Ra.transpose() * Rb;
    //   Eigen::Vector3d res_p = pb - pa;
    //   rr.setZero();
    //   rr.block<3, 1>(0, 0) = Log(res_r);
    //   rr.block<3, 1>(3, 0) = Ra.transpose() * res_p - pab;

    //   residual1 += rr.squaredNorm();
    // }

    // rr.setZero();
    // Eigen::Matrix3d res_r = x_ab[0].R.transpose() * x_stats[0].R;
    // rr.block<3, 1>(0, 0) = Log(res_r);
    // rr.block<3, 1>(3, 0) = x_stats[0].p - x_ab[0].p;
    // residual1 += rr.squaredNorm();

    // residual1 *= (vel_coef * 0.5);

    voxhess.evaluate_only_residual(x_stats, residual2);
    if(is_avg) return residual2 / voxhess.plvec_voxels.size();
    return (residual1 + residual2);
  }

  void remove_outlier(vector<IMUST>& x_stats, VOX_HESS& voxhess, double ratio)
  {
    std::vector<double> residuals = voxhess.evaluate_residual(x_stats);
    std::sort(residuals.begin(), residuals.end()); // sort in ascending order
    // std::ofstream file;
    // file.open("/home/sam/Documents/Laser_Registration_Dataset/haupt/tmp/residual.txt", std::ofstream::trunc);
    // file.close();
    // file.open("/home/sam/Documents/Laser_Registration_Dataset/haupt/tmp/residual.txt", std::ofstream::app);
    // for(size_t i = 0; i < residuals.size(); i++) file << residuals[i] << (i==(residuals.size()-1)?"":"\n");
    // file.close();
    double threshold = residuals[std::floor((1-ratio)*voxhess.plvec_voxels.size())-1];
    int reject_num = std::floor(ratio * voxhess.plvec_voxels.size());
    std::cout << "vox_num before " << voxhess.plvec_voxels.size();
    std::cout << ", reject threshold " << std::setprecision(3) << threshold << ", rejected " << reject_num;
    voxhess.remove_residual(x_stats, threshold, reject_num);
    std::cout << ", vox_num after " << voxhess.plvec_voxels.size() << std::endl;
  }

  void damping_iter(vector<IMUST>& x_stats, VOX_HESS& voxhess, double& residual,
                    Eigen::MatrixXd& Hess_, size_t& mem_cost)
  {
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng),
                    HessuD(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng), new_dxi(jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp;

    vector<IMUST> x_ab(win_size);
    x_ab[0] = x_stats[0];
    for(int i=1; i<win_size; i++)
    {
      x_ab[i].p = x_stats[i-1].R.transpose() * (x_stats[i].p - x_stats[i-1].p);
      x_ab[i].R = x_stats[i-1].R.transpose() * x_stats[i].R;
    }

    double hesstime = 0;
    double solvtime = 0;
    size_t max_mem = 0;
    double loop_num = 0;
    for(int i = 0; i < 10; i++)
    {
      if(is_calc_hess)
      {
        double tm = ros::Time::now().toSec();
        residual1 = divide_thread(x_stats, voxhess, x_ab, Hess, JacT);
        hesstime += ros::Time::now().toSec() - tm;
        printf("Avg. Hessian time: %lf ", hesstime/(loop_num+1));
      }

      // 
      // D.diagonal() = Hess.diagonal();
      // dxi = (Hess + u*D).colPivHouseholderQr().solve(-JacT);
      // printf("solve time cost %f\n",ros::Time::now().toSec() - tm);
      // double relative_err = ((Hess + u*D)*dxi + JacT).norm()/JacT.norm();
      // double absolute_err = ((Hess + u*D)*dxi + JacT).norm();
      // cout<<setiosflags(ios::fixed)<<setprecision(24);
      // std::cout<<"relative error "<<relative_err<<std::endl;
      // std::cout<<"absolute error "<<absolute_err<<std::endl;
      // 

      D.diagonal() = Hess.diagonal();
      // HessuD = Hess + u*D;
      HessuD = u*D;

      // cv::Mat full_hess(cv::Size(win_size, win_size), CV_8UC1, cv::Scalar(255));
      // for(int i = 0; i < win_size; i++)
      //   for(int j = 0; j < win_size; j++)
      //     if(Hess(6*i, 6*j) != 0)
      //       full_hess.at<uchar>(i, j) = 0;
      // cv::imwrite("/media/sam/T7/KITTI/07/full_hess.jpg", full_hess);
      
      // cv::Mat hess(cv::Size(win_size, win_size), CV_8UC1, cv::Scalar(255));
      for(int j = 0; j < win_size; j++)
        if(j%GAP == 0)
          for(int k = 0; k < WIN_SIZE-1; k++)
          {
            if(j+k+1 > win_size) break;
            // hess.at<uchar>(j+k, j+k+1) = 0;
            // hess.at<uchar>(j+k+1, j+k) = 0;
            HessuD.block(6*(j+k), 6*(j+k+1), 6, 6) = Hess.block(6*(j+k), 6*(j+k+1), 6, 6);
            HessuD.block(6*(j+k+1), 6*(j+k), 6, 6) = Hess.block(6*(j+k+1), 6*(j+k), 6, 6);
          }
      
      int low_pose_size = win_size/GAP;
      for(int j = 0; j < low_pose_size; j++)
        if(j%GAP == 0)
          for(int k = 0; k < WIN_SIZE-1; k++)
          {
            if((j+k+1)*GAP > win_size) break;
            // hess.at<uchar>((j+k)*GAP, (j+k+1)*GAP) = 0;
            // hess.at<uchar>((j+k+1)*GAP, (j+k)*GAP) = 0;
            HessuD.block(6*(j+k)*GAP, 6*(j+k+1)*GAP, 6, 6) = Hess.block(6*(j+k)*GAP, 6*(j+k+1)*GAP, 6, 6);
            HessuD.block(6*(j+k+1)*GAP, 6*(j+k)*GAP, 6, 6) = Hess.block(6*(j+k+1)*GAP, 6*(j+k)*GAP, 6, 6);
          }

      int up_pose_size = low_pose_size/GAP;
      for(int j = 0; j < up_pose_size-1; j++)
      {
        if((j+1)*pow(GAP, 2) > win_size) break;
        // hess.at<uchar>(j*pow(GAP, 2), (j+1)*pow(GAP, 2)) = 0;
        // hess.at<uchar>((j+1)*pow(GAP, 2), j*pow(GAP, 2)) = 0;
        HessuD.block(6*j*pow(GAP, 2), 6*(j+1)*pow(GAP, 2), 6, 6) = Hess.block(6*j*pow(GAP, 2), 6*(j+1)*pow(GAP, 2), 6, 6);
        HessuD.block(6*(j+1)*pow(GAP, 2), 6*j*pow(GAP, 2), 6, 6) = Hess.block(6*(j+1)*pow(GAP, 2), 6*j*pow(GAP, 2), 6, 6);
      }
      // cv::imwrite("/media/sam/T7/KITTI/07/simp_hess.jpg", hess);

      double tm = ros::Time::now().toSec();
      double t1 = ros::Time::now().toSec();
      Eigen::SparseMatrix<double> A1_sparse(jac_leng, jac_leng);
      std::vector<Eigen::Triplet<double>> tripletlist;
      for(int a = 0; a < jac_leng; a++)
        for(int b = 0; b < jac_leng; b++)
          if(HessuD(a, b) != 0)
          {
            tripletlist.push_back(Eigen::Triplet<double>(a, b, HessuD(a, b)));
            //A1_sparse.insert(a, b) = HessuD(a, b);
          }
      A1_sparse.setFromTriplets(tripletlist.begin(), tripletlist.end());
      A1_sparse.makeCompressed();
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
      Solver_sparse.compute(A1_sparse);
      size_t temp_mem = check_mem();
      if(temp_mem > max_mem) max_mem = temp_mem;
      dxi = Solver_sparse.solve(-JacT);
      temp_mem = check_mem();
      if(temp_mem > max_mem) max_mem = temp_mem;
      solvtime += ros::Time::now().toSec() - tm;
      printf("Avg. solve time: %lf\n", solvtime/(loop_num+1));
      // new_dxi = Solver_sparse.solve(-JacT);
      // printf("new solve time cost %f\n",ros::Time::now().toSec() - t1);
      // relative_err = ((Hess + u*D)*dxi + JacT).norm()/JacT.norm();
      // absolute_err = ((Hess + u*D)*dxi + JacT).norm();
      // std::cout<<"relative error "<<relative_err<<std::endl;
      // std::cout<<"absolute error "<<absolute_err<<std::endl;
      // std::cout<<"delta x\n"<<(new_dxi-dxi).transpose()/dxi.norm()<<std::endl;

      x_stats_temp = x_stats;
      for(int j = 0; j < win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);
      }

      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
      #ifdef AVG_THR
      residual2 = only_residual(x_stats_temp, voxhess, x_ab, true);
      #else
      residual2 = only_residual(x_stats_temp, voxhess, x_ab);
      #endif
      residual = only_residual(x_stats_temp, voxhess, x_ab, true);

      q = (residual1-residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %lf q: %lf %lf %lf\n",
      //        i, residual1, residual2, u, v, q/q1, q1, q);
      loop_num = i+1;
      if(hesstime/loop_num > 1) printf("Avg. Hessian time: %lf ", hesstime/loop_num);
      if(solvtime/loop_num > 1) printf("Avg. solve time: %lf\n", solvtime/loop_num);
      if(double(max_mem/1048576.0) > 1.0) printf("Max mem: %lf\n", double(max_mem/1048576.0));

      if(q > 0)
      {
        x_stats = x_stats_temp;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
      }
      #ifdef AVG_THR
      if((fabs(residual1-residual2)/residual1)<0.05)
      {
        Hess_ = Hess;
        if(mem_cost < max_mem) mem_cost = max_mem;
        break;
      }
      #else
      if(fabs(residual1-residual2)<1e-9) break;
      #endif
    }
  }

  size_t check_mem()
  {
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while(fgets(line, 128, file) != nullptr)
    {
      if(strncmp(line, "VmRSS:", 6) == 0)
      {
        int len = strlen(line);

        const char* p = line;
        for(; std::isdigit(*p) == false; ++p){}

        line[len - 3] = 0;
        result = atoi(p);

        break;
      }
    }
    fclose(file);

    return result;
  }
};

#endif