#include "deann.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <iostream>
#include <random>



/**
 * @file
 * Python bindings for the C++ procedures
 */

namespace py = pybind11;

using namespace deann;
using std::string;
using std::to_string;
using std::invalid_argument;
using std::unique_ptr;
using std::make_unique;
using std::optional;
using std::mt19937;
using std::nullopt;
using std::random_device;



namespace {
  enum class Dtype {
    FLOAT32, FLOAT64, NONE
  };



  static bool hasDtype(const py::array& X, Dtype dt) {
    switch(dt) {
    case Dtype::FLOAT32:
      return py::isinstance<py::array_t<float>>(X);
      break;
    case Dtype::FLOAT64:
      return py::isinstance<py::array_t<double>>(X);
      break;
    case Dtype::NONE:
      return false;
      break;
    }
    assert(false && "This line should never be reached");
    return 0; // to suppress warnings
  }



  template<typename T>
  static bool hasDtype(const py::array& X) {
    return py::isinstance<py::array_t<T>>(X);
  }



  static bool isCStyleContiguous(const py::array& X) {
    return X.flags() & py::array::c_style;
  }



  static Dtype extractDtype(const py::array& X) {
    if (py::isinstance<py::array_t<float>>(X) &&
	!py::isinstance<py::array_t<double>>(X))
      return Dtype::FLOAT32;
    else if (!py::isinstance<py::array_t<float>>(X) &&
	py::isinstance<py::array_t<double>>(X))
      return Dtype::FLOAT64;
    else 
      return Dtype::NONE;
  }



  static_assert(std::is_same<KdeEstimator::PseudoRandomNumberGenerator::ValueType, uint32_t>::value);

  class KdeEstimatorWrapper {
  public:
    typedef KdeEstimator::PseudoRandomNumberGenerator::ValueType SeedType;

    
    
    KdeEstimatorWrapper(double bandwidth, const string& kernel,
			optional<uint32_t> randomSeed = nullopt) {
      resetSeed(randomSeed);
      setBandwidth(bandwidth);
      setKernel(kernel);
    }



    virtual void resetParameters(optional<uint32_t> param1 = nullopt,
                                 optional<uint32_t> param2 = nullopt) = 0;



    void resetSeed(optional<uint32_t> randomSeed = nullopt) {
      rngSeed = randomSeed;
      if (est.get())
	est->resetSeed(rngSeed);
    }



    void setBandwidth(double bandwidth) {
      if (bandwidth <= 0)
	throw invalid_argument("Bandwidth must be positive (got " +
			       to_string(h) + ")");
      h = bandwidth;
    }



    void setKernel(const string& kernel) {
      if (kernel == "exponential")
	K = Kernel::EXPONENTIAL;
      else if (kernel == "gaussian")
	K = Kernel::GAUSSIAN;
      else if (kernel == "laplacian")
	K = Kernel::LAPLACIAN;
      else
	throw invalid_argument("Supported kernels are ``exponential'', "
			       "``gaussian'', and ``laplacian'' (got: " +
			       kernel + ")");
    }



    virtual ~KdeEstimatorWrapper() {
    }



    inline double getBandwidth() const {
      return h;
    }



    inline Kernel getKernel() const {
      return K;
    }
    


    inline Dtype getDtype() const {
      return dtype;
    }



    inline int getD() const {
      return d;
    };

    

    template<typename T>
    inline const KdeEstimatorT<T>& getEstimator() const {
      return dynamic_cast<const KdeEstimatorT<T>&>(*est);
    }



    inline KdeEstimator* getEstimatorPtr() {
      return est.get();
    }


    
    void fit(const py::array& X) {
      if (!isCStyleContiguous(X))
        throw invalid_argument("The dataset must be a C-style contiguous "
                               "array");
      
      auto inf = X.request();
     
      if (inf.ndim != 2)
	throw invalid_argument("Dataset must have ndim=2 (got ndim=" +
			       to_string(inf.ndim) + ")");
      Dtype dt = extractDtype(X);
      if (dt != Dtype::FLOAT32 && dt != Dtype::FLOAT64)
	throw invalid_argument("Dataset dtype must be either float32 "
			       "or float64, got " +
			       py::cast<std::string>(py::str(X.dtype())));
      if (!(inf.shape[0] > 0))
	throw invalid_argument("Dataset must have a positive number of rows");
      if (!(inf.shape[1] > 0))
	throw invalid_argument("Dataset must have a positive number of cols");

      int n = inf.shape[0];
      int d = inf.shape[1];
      est.reset(construct(dt));
      if (dt == Dtype::FLOAT64)
	dynamic_cast<KdeEstimatorT<double>*>(est.get())->
	  fit(n, d, reinterpret_cast<double*>(inf.ptr));
      else if (dt == Dtype::FLOAT32)
	dynamic_cast<KdeEstimatorT<float>*>(est.get())->
	  fit(n, d, reinterpret_cast<float*>(inf.ptr));
      
      this->X = X;
      this->n = n;
      this->d = d;

      dtype = dt;
      
    }




    py::tuple query(const py::array& Q) {
      if (!isCStyleContiguous(Q))
        throw invalid_argument("The queries must be in a C-style contiguous "
                               "array");

      if (dtype == Dtype::FLOAT32)
	return queryImpl<float>(Q, getEstimator<float>());
      else if (getDtype() == Dtype::FLOAT64)
	return queryImpl<double>(Q, getEstimator<double>());
      throw invalid_argument("The object has not been initialized (fit must be "
			     "called before queries can be made)");
      return py::tuple();
    }



  protected:
    inline optional<SeedType> getRngSeed() const {
      static_assert(std::is_same<SeedType, decltype(random_device()())>::value);
      return rngSeed;
    }

    

  private:
    template<typename T>
    py::tuple queryImpl(const py::array& Q, const KdeEstimatorT<T>& est) {
      if (!hasDtype<T>(Q))
       	throw invalid_argument("Expected queries to have dtype " +
      			       py::cast<string>(py::str(py::dtype::of<T>())) +
      			       ", but got " +
      			       py::cast<string>(py::str(Q.dtype())));
      auto inf = Q.request();
      if (inf.ndim > 2)
      	throw invalid_argument("ndim=1 or ndim=2 assumed, but got ndim=" +
      			       to_string(inf.ndim));

      int m = inf.ndim == 2 ? inf.shape[0] : 1;
      
      if (m < 1)
      	throw invalid_argument("expected a positive number of rows but got " +
      			       to_string(m));
      int gotD;
      if (inf.ndim == 2)
	gotD = inf.shape[1];
      else
	gotD = inf.shape[0];
      
      if (gotD != d)
	throw invalid_argument("dimension mismatch (got: " + to_string(gotD) +
			       + ", but expected: " + to_string(d) + ")");

      py::array S(py::dtype::of<int>(), m);
      int* Sptr = reinterpret_cast<int*>(S.request().ptr);
      
      py::array Z(Q.dtype(), m);
      T* Zptr = reinterpret_cast<T*>(Z.request().ptr);
      
      const T* Qptr = reinterpret_cast<const T*>(inf.ptr);

      est.query(m, d, Qptr, Zptr, Sptr);

      return py::make_tuple(Z, S);
    }


    
    virtual KdeEstimator* construct(Dtype dt) = 0;


    
    double h = 0;
    Kernel K = Kernel::EXPONENTIAL;
    optional<SeedType> rngSeed = std::nullopt;
    py::array X;
    int n = 0;
    int d = 0;
    Dtype dtype = Dtype::NONE;
    unique_ptr<KdeEstimator> est;
  };


  
  class NaiveKdeWrapper : public KdeEstimatorWrapper {
  public:
    NaiveKdeWrapper(double h, const string& kernel) :
      KdeEstimatorWrapper(h, kernel) {
    }  
 

    
    void resetParameters(optional<uint32_t> param1 = nullopt,
                         optional<uint32_t> param2 = nullopt) override {
      if (param1 || param2)
        throw invalid_argument("Tried to reset parameters on NaiveKde, but the "
                               "class does not support any parameters.");
    }

    

  private:
    KdeEstimator* construct(Dtype dt) override {
      if (dt == Dtype::FLOAT64) {
	return new NaiveKde<double>(getBandwidth(), getKernel());
      }
      else if (dt == Dtype::FLOAT32) {
	return new NaiveKde<float>(getBandwidth(), getKernel());
      }
      else {
	throw invalid_argument("Initialization failed for unknown reason, please debug.");
	return nullptr;
      }
    }
  };



  class RandomSamplingWrapper : public KdeEstimatorWrapper {
  public:
    RandomSamplingWrapper(double h, const string& kernel, int samples,
			  optional<uint32_t> randomSeed = nullopt) :
      KdeEstimatorWrapper(h, kernel, randomSeed) {
      resetParameters(samples);
    }



    void resetParameters(optional<uint32_t> param1 = nullopt,
                         optional<uint32_t> param2 = nullopt) override {
      if (param1) {
        int newSamples = *param1;
        if (newSamples < 1)
          throw invalid_argument("Random samples must be positive (got: " +
                                 to_string(newSamples) + ")");
        randomSamples = newSamples;
        KdeEstimator* ptr = getEstimatorPtr();
        if (ptr)
          ptr->resetParameters(newSamples);
      }
      if (param2)
        throw invalid_argument("Tried to reset two parameters for the "
                               "RandomSampling class, but the class only "
                               "supports one parameter (the number of random "
                               "samples)");
    }


    
  private:
    KdeEstimator* construct(Dtype dt) override {
      if (dt == Dtype::FLOAT64) {
	return new RandomSampling<double>(getBandwidth(), getKernel(),
					  randomSamples, getRngSeed());
      }
      else if (dt == Dtype::FLOAT32) {
	return new RandomSampling<float>(getBandwidth(), getKernel(),
					 randomSamples, getRngSeed());
      }
      else {
	throw invalid_argument("Initialization failed for unknown reason, please debug.");
	return nullptr;
      }
    }

    

    int randomSamples = 0;
  };



  class RandomSamplingPermutedWrapper : public KdeEstimatorWrapper {
  public:
    RandomSamplingPermutedWrapper(double h, const string& kernel, int samples,
				  optional<uint32_t> randomSeed = nullopt) :
      KdeEstimatorWrapper(h, kernel, randomSeed) {
      resetParameters(samples);
      if (kernel != "exponential" && kernel != "gaussian")
	throw std::invalid_argument("Only exponential and gaussian kernels are "
				    "supported by this class.");
    }



    void resetParameters(optional<uint32_t> param1 = nullopt,
                         optional<uint32_t> param2 = nullopt) override {
      if (param1) {
        int newSamples = *param1;
        if (newSamples < 1)
          throw invalid_argument("Random samples must be positive (got: " +
                                 to_string(newSamples) + ")");
        randomSamples = newSamples;
        KdeEstimator* ptr = getEstimatorPtr();
        if (ptr)
          ptr->resetParameters(newSamples);
      }
      if (param2) 
        throw invalid_argument("Tried to reset two parameters for the "
                               "RandomSamplingPermuted class, but the class "
                               "only supports one parameter (the number of "
                               "random samples)");
    }


    
  private:
    KdeEstimator* construct(Dtype dt) override {
      if (dt == Dtype::FLOAT64) {
	return new RandomSamplingPermuted<double>(getBandwidth(), getKernel(),
						  randomSamples, getRngSeed());
      }
      else if (dt == Dtype::FLOAT32) {
	return new RandomSamplingPermuted<float>(getBandwidth(), getKernel(),
						 randomSamples, getRngSeed());
      }
      else {
	throw invalid_argument("Initialization failed for unknown reason, "
			       "please debug.");
	return nullptr;
      }
    }

    

    int randomSamples = 0;
  };



  class LinearScanWrapper {
  public:
    LinearScanWrapper(const string& metric) :
      metric(metric) {
      if (metric != "euclidean" && metric != "taxicab")
	throw invalid_argument("Supported metrics are ``euclidean'', "
			       "and ``taxicab'' (got: " +
			       metric + ")");
    }



    void fit(const py::array& X) {
      if (!isCStyleContiguous(X))
        throw invalid_argument("The dataset must be a C-style contiguous "
                               "array");

      auto inf = X.request();
      if (inf.ndim != 2)
      	throw invalid_argument("Dataset must have ndim=2 (got ndim=" +
      			       to_string(inf.ndim) + ")");
      if (!hasDtype(X, Dtype::FLOAT32) && !hasDtype(X, Dtype::FLOAT64))
      	throw invalid_argument("Dataset dtype must be either float32 "
      			       "or float64, got " +
      			       py::cast<std::string>(py::str(X.dtype())));
      if (hasDtype(X, Dtype::FLOAT32) && hasDtype(X, Dtype::FLOAT64))
      	throw invalid_argument("Conflicting dtypes determined, this should not happen.");
      if (!(inf.shape[0] > 0))
      	throw invalid_argument("Dataset must have a positive number of rows");
      if (!(inf.shape[1] > 0))
      	throw invalid_argument("Dataset must have a positive number of cols");

      lsf.reset(nullptr);
      lsd.reset(nullptr);

      this->X = X;
      n = inf.shape[0];
      d = inf.shape[1];
      Metric M = (metric == "euclidean" ? Metric::EUCLIDEAN :
		  metric == "taxicab" ? Metric::TAXICAB :
		  Metric::EUCLIDEAN);
      if (hasDtype(X, Dtype::FLOAT64)) {
      	dtype = Dtype::FLOAT64;
      	lsd = make_unique<LinearScan<double>>(M);
      	lsd->fit(n, d, reinterpret_cast<double*>(inf.ptr));
      }
      else if (hasDtype(X, Dtype::FLOAT32)) {
      	dtype = Dtype::FLOAT32;
      	lsf = make_unique<LinearScan<float>>(M);
      	lsf->fit(n, d, reinterpret_cast<float*>(inf.ptr));
      }
      else {
      	throw invalid_argument("Initialization failed for unknown reason, please debug.");
      }
    }



    py::tuple query(const py::array& Q, int k) {
      if (!isCStyleContiguous(Q))
        throw invalid_argument("The queries must be in a C-style contiguous "
                               "array");

      if (dtype == Dtype::NONE)
	throw invalid_argument("The object has not been initialized (fit must be "
			       "called before queries can be made)");

      if (k < 1)
	throw invalid_argument("Must query at least one neighbor (got: " +
			       to_string(k) + ")");
      if (k > n)
	throw invalid_argument("Cannot query more than " + to_string(n) +
			       " points");
      if (dtype == Dtype::FLOAT32)
      	return queryImpl<float>(Q, *lsf, k);
      else if (dtype == Dtype::FLOAT64)
      	return queryImpl<double>(Q, *lsd, k);
      throw invalid_argument("Something went wrong which shouldn't be "
			     "possible");
      return py::tuple();
    }


    
  private:
    template<typename T>
    py::tuple queryImpl(const py::array& Q, const LinearScan<T>& ls, int k) {
      if (!hasDtype<T>(Q))
       	throw invalid_argument("Expected queries to have dtype " +
      			       py::cast<string>(py::str(py::dtype::of<T>())) +
      			       ", but got " +
      			       py::cast<string>(py::str(Q.dtype())));
      auto inf = Q.request();
      
      if (inf.ndim > 2)
      	throw invalid_argument("ndim=1 or ndim=2 assumed, but got ndim=" +
      			       to_string(inf.ndim));
      int m = inf.ndim == 1 ? 1 : inf.shape[0];
      
      if (m < 1)
      	throw invalid_argument("expected a positive number of rows but got " +
      			       to_string(m));
      
      if ((inf.ndim == 2 && inf.shape[1] != d) ||
	  (inf.ndim == 1 && inf.shape[0] != d))
      	throw invalid_argument("dimension mismatch");
      
      py::array N(py::dtype::of<int>(), {m,k});
      int* Nptr = reinterpret_cast<int*>(N.request().ptr);
      py::array D(py::dtype::of<T>(), {m,k});
      T* Dptr = reinterpret_cast<T*>(D.request().ptr);
      const T* Qptr = reinterpret_cast<const T*>(inf.ptr);
      py::array S(py::dtype::of<int>(), m);
      int* Sptr = reinterpret_cast<int*>(S.request().ptr);
      ls.query(m, d, k, Qptr, Nptr, Dptr, Sptr);
      return py::make_tuple(D, N, S);
    }
    
    

    string metric;
    py::array X;
    int n = 0;
    int d = 0;
    Dtype dtype = Dtype::NONE;
    unique_ptr<LinearScan<float>> lsf;
    unique_ptr<LinearScan<double>> lsd;
  };

  

  class AnnObject {
  public:
    explicit AnnObject(py::object ann) : ann(ann) {
      if (!py::hasattr(ann, "query"))
        throw invalid_argument("The ann object provided does not have a "
                               "query function.");
    }
    
    
    
    template<typename T>
    void query(int d, int k, const T* Q, int* N, T* D, int* S) const {
      py::capsule capsule(Q, [](void*) { });
      py::array_t<T> arr({d}, {sizeof(T)}, Q, capsule);
      
      py::object resObject = ann.attr("query")(arr, k);
      py::array nn;
      py::array dists;
      py::array samples;
      bool hasDists = false;
      bool hasSamples = false;
      if (py::isinstance<py::tuple>(resObject)) {
        py::tuple tup = py::cast<py::tuple>(resObject);
        static_assert(std::is_same<size_t,decltype(py::len(tup))>::value);
        size_t tupLen = py::len(tup);
        if (py::len(tup) != 3)
          throw invalid_argument("Got a tuple of length " + to_string(tupLen)
                                 + ", but 3 was expected.");
        if (!py::isinstance<py::array>(tup[0]))
          throw invalid_argument("Tuple element 0 is not an np.ndarray");
        if (!py::isinstance<py::array>(tup[1]))
          throw invalid_argument("Tuple element 1 is not an np.ndarray");
        if (!py::isinstance<py::array>(tup[2]))
          throw invalid_argument("Tuple element 2 is not an np.ndarray");
        samples = tup[2];
        nn = tup[1];
        dists = tup[0];
        hasDists = true;
        hasSamples = true;
      }
      else if (py::isinstance<py::array>(resObject)){
        nn = py::cast<py::array>(resObject);
      }
      else {
        throw invalid_argument("Expected a tuple of two np.ndarrays, or a "
                               "single np.ndarray, but got " +
                               py::cast<string>(py::str(resObject)) +
                               " instead");
      }
      
      py::buffer_info nn_inf = nn.request();
      if (nn.ndim() > 2 || (nn.ndim() == 2 && nn_inf.shape[0] != 1))
        throw invalid_argument("ANN object returned an array of ndim=" +
                               to_string(nn.ndim()) + ", but ndim=1 was "
                               "expected, or ndim=2 but with first dimension "
                               "of length 1");
      
      py::buffer_info dists_inf;
      if (hasDists) {
        dists_inf = dists.request();
        if (nn.ndim() != dists.ndim() ||
            nn_inf.shape[0] != dists_inf.shape[0] ||
            (nn.ndim() == 2 && nn_inf.shape[1] != dists_inf.shape[1]))
          throw invalid_argument("ANN object two arrays whose shapes do not "
                                 "match (nn indices and distances do not "
                                 "match)");
      }	
      
      int m = nn.ndim() == 1 ? nn_inf.shape[0] : nn_inf.shape[1];
      if (m > k)
        throw invalid_argument("Requested " + to_string(k) + " neighbors, "
                               "but " + to_string(m) + " were returned.");
      
      
      py::buffer_info samples_inf;
      if (hasSamples) {
        samples_inf = samples.request();
        if (!(samples.ndim() == 1 && samples_inf.shape[0] == 1) &&
            !(samples.ndim() == 2 && samples_inf.shape[0] == 1 && samples_inf.shape[1] == 1))
          throw invalid_argument("Invalid number of samples returned by the ANN object");
      }
      
      if (hasDtype<int32_t>(nn)) {
        int32_t* p = reinterpret_cast<int32_t*>(nn_inf.ptr);
        for (int i = 0; i < m; ++i)
          N[i] = p[i];
        for (int i = m; i < k; ++i)
          N[i] = -1;
      }
      else if (hasDtype<int64_t>(nn)) {
        int64_t* p = reinterpret_cast<int64_t*>(nn_inf.ptr);
        for (int i = 0; i < m; ++i)
          N[i] = p[i];
        for (int i = m; i < k; ++i)
          N[i] = -1;
      }
      else {
        throw invalid_argument("Only int32 and int64 types are supported for "
                               "ANN indices.");
      }
      
      if (hasDists) {
        if (hasDtype<float>(dists)) {
          float* p = reinterpret_cast<float*>(dists_inf.ptr);
          for (int i = 0; i < m; ++i)
            D[i] = p[i];
          for (int i = m; i < k; ++i)
            D[i] = -1;
        }
        else if (hasDtype<double>(dists)) {
          double* p = reinterpret_cast<double*>(dists_inf.ptr);
          for (int i = 0; i < m; ++i)
            D[i] = p[i];
          for (int i = m; i < k; ++i)
            D[i] = -1;
        }
        else {
          throw invalid_argument("Only float32 and float64 types are supported for "
                                 "ANN distances.");
        }
      }
      else {
        for (int i = 0; i < k; ++i)
          D[i] = -1;
      }
      
      if (hasSamples) {
        if (hasDtype<int32_t>(samples)) {
          *S = *reinterpret_cast<int32_t*>(samples_inf.ptr);
        }
        else if (hasDtype<int64_t>(samples)) {
          *S = *reinterpret_cast<int64_t*>(samples_inf.ptr);
        }
        else {
          throw invalid_argument("Only int32 and int64 types are supported for sample counts");
        }
      }
      else {
        *S = -1;
      }
    }
    
    
    
  private:
    py::object ann;
  };



  class AnnEstimatorWrapper : public KdeEstimatorWrapper {
  public:
    AnnEstimatorWrapper(double h, const string& kernel, int nearNeighbors,
			int randomSamples, py::object ann, 
			optional<uint32_t> randomSeed = nullopt) :
      KdeEstimatorWrapper(h, kernel, randomSeed), annObject(ann) {
      resetParameters(nearNeighbors, randomSamples);
    }

    

    void resetParameters(optional<uint32_t> param1 = nullopt,
                         optional<uint32_t> param2 = nullopt) override {
      int newNeighbors = param1 ? *param1 : nearNeighbors;
      int newSamples = param2 ? *param2 : randomSamples;
      if (newNeighbors < 0)
	throw invalid_argument("Near neighbors must be non-negative (got: " +
			       to_string(nearNeighbors) + ")");
      if (newSamples < 0)
	throw invalid_argument("Random samples must be non-negative (got: " +
			       to_string(randomSamples) + ")");
      nearNeighbors = newNeighbors;
      randomSamples = newSamples;
      KdeEstimator* ptr = getEstimatorPtr();
      if (ptr)
	ptr->resetParameters(newNeighbors, newSamples);
    }

    
    
  private:
    KdeEstimator* construct(Dtype dt) override {
      if (dt == Dtype::FLOAT64) {
      	return new AnnEstimator<double,AnnObject>(getBandwidth(),
						  getKernel(),
						  nearNeighbors,
						  randomSamples,
						  &annObject,
						  getRngSeed());
      }
      else if (dt == Dtype::FLOAT32) {
      	return new AnnEstimator<float,AnnObject>(getBandwidth(),
						 getKernel(),
						 nearNeighbors,
						 randomSamples,
						 &annObject,
						 getRngSeed());
      }
      else {
      	throw invalid_argument("Initialization failed for unknown reason, please debug.");
	return nullptr;
      }
    }
 
   

    int nearNeighbors = 0;
    int randomSamples = 0;
    AnnObject annObject;
  };



  class AnnEstimatorPermutedWrapper : public KdeEstimatorWrapper {
  public:
    AnnEstimatorPermutedWrapper(double h, const string& kernel, int nearNeighbors,
                                int randomSamples, py::object ann, 
                                optional<uint32_t> randomSeed = nullopt) :
      KdeEstimatorWrapper(h, kernel, randomSeed), annObject(ann) {
      resetParameters(nearNeighbors, randomSamples);
    }

    

    void resetParameters(optional<uint32_t> param1 = nullopt,
                         optional<uint32_t> param2 = nullopt) override {
      int newNeighbors = param1 ? *param1 : nearNeighbors;
      int newSamples = param2 ? *param2 : randomSamples;
      if (newNeighbors < 0)
	throw invalid_argument("Near neighbors must be non-negative (got: " +
			       to_string(nearNeighbors) + ")");
      if (newSamples < 0)
	throw invalid_argument("Random samples must be non-negative (got: " +
			       to_string(randomSamples) + ")");
      nearNeighbors = newNeighbors;
      randomSamples = newSamples;
      KdeEstimator* ptr = getEstimatorPtr();
      if (ptr)
	ptr->resetParameters(newNeighbors, newSamples);
    }

    
    
  private:
    KdeEstimator* construct(Dtype dt) override {
      if (dt == Dtype::FLOAT64) {
      	return new AnnEstimatorPermuted<double,AnnObject>(getBandwidth(),
                                                          getKernel(),
                                                          nearNeighbors,
                                                          randomSamples,
                                                          &annObject,
                                                          getRngSeed());
      }
      else if (dt == Dtype::FLOAT32) {
      	return new AnnEstimatorPermuted<float,AnnObject>(getBandwidth(),
                                                         getKernel(),
                                                         nearNeighbors,
                                                         randomSamples,
                                                         &annObject,
                                                         getRngSeed());
      }
      else {
      	throw invalid_argument("Initialization failed for unknown reason, please debug.");
	return nullptr;
      }
    }
 
   

    int nearNeighbors = 0;
    int randomSamples = 0;
    AnnObject annObject;
  };
}



PYBIND11_MODULE(deann, m) {
  m.doc() = "DEANN Python Bindings";

  py::class_<KdeEstimatorWrapper>(m, "KdeEstimator",
                                  R"pydoc(Base class for KDE estimators.

This class provides the common interface for estimating Kernel Density Estimate
(KDE) values. In general, the KDE is defined as the following sum:

.. math::
    \mathrm{KDE}_X(q) = \frac{1}{n} \sum_{i=0}^{n-1} K_h(q,x_i)

with respect to a dataset of vectors :math:`\{x_0, x_1, ..., x_{n-1}\}` specified
as the rows of a :math:`n\times d` matrix, using a kernel function :math:`K_h` 
with a bandwidth parameter :math:`h > 0`.)pydoc")
    .def("fit", &KdeEstimatorWrapper::fit, R"pydoc(Fit a dataset into the data 
structure.

:param dataset: A matrix of shape :math:`n\times d`, dtype must be float64 or 
    float32.
:type dataset: :class:`np.ndarray`)pydoc",
	 py::arg("dataset"))
    .def("query", &KdeEstimatorWrapper::query, R"pydoc(Query the estimator for 
:math:`N` query vectors.

:param queries: A :math:`N\times d` matrix of vectors to query, or a single query
    vector of length :math:`d`. The dimension :math:`d` must match that of the 
    dataset.
:type queries: :class:`np.ndarray`
:return: A tuple :math:`(Z,S)`, the elements of which are vectors vector of 
    length :math:`N`. The elements of the vector :math:`Z` correspond to the KDE
    estimates of each row of the queries. The elements of :math:`S` tell how 
    many samples were looked at to compute the respective query if this 
    information is available; otherwise the elements are negative.
:rtype: :class:`tuple`)pydoc",
	 py::arg("queries"))
    .def("reset_seed", &KdeEstimatorWrapper::resetSeed,
	 R"pydoc(Reset the random number seed for the estimator.

This function can be used to repeat experiments with the same random choices.

:param seed: Random number seed or None for unpredictable initialization.
:type seed: int, optional)pydoc", py::arg("seed") = nullopt)
    .def("reset_parameters", &KdeEstimatorWrapper::resetParameters,
         R"pydoc(Reset the parameters for the estimator.

You can supply up to two estimator-specific parameters, or None, to set one or 
two parameters of the estimator. Whatever this means is determined by subclass.
An exception will be raised if the combination is not supported.

:param param1: First parameter.
:type param1: int, optional
:param param2: Second parameter.
:type param2: int, optional)pydoc",
	 py::arg("param1") = nullopt, py::arg("param2") = nullopt);
  
  py::class_<NaiveKdeWrapper,KdeEstimatorWrapper>(m, "NaiveKde",
                                                  R"pydoc(Computes the exact KDE value naively.

This class provides exact KDE computation. For the Euclidean metric, matrix 
multiplication is used to speed up the computation.)pydoc")
    .def(py::init<double, const string&>(),
	 R"pydoc(Constructor.

:param bandwidth: The bandwidth :math:`h>0`.
:type bandwidth: float
:param kernel: A string to select the kernel. Allowed values are ``'gaussian'``, 
    ``'exponential'``, and ``'laplacian'``. These choices correspond to the 
    following kernels:

    | ``'gaussian'``: :math:`K_h(q,x) = \exp\left(-\frac{||x-q||_2^2}{2h^2}\right)`
    | ``'exponential'``: :math:`K_h(q,x) = \exp\left(-\frac{||x-q||_2}{h}\right)`
    | ``'laplacian'``: :math:`K_h(q,x) = \exp\left(-\frac{||x-q||_1}{h}\right)`
:type kernel: str)pydoc",
	 py::arg("bandwidth"), py::arg("kernel"));

  py::class_<RandomSamplingWrapper,KdeEstimatorWrapper>(m, "RandomSampling",
                                                        R"pydoc(The Random Sampling estimator.

Computes the unbiased estimate :math:`Z = \frac{1}{m}\sum_{x'\in X'} K_h(q,x')` 
where the sequence :math:`X'=(x'_0,x'_1,\ldots,x'_{m-1})` is chosen uniformly 
and independently at random.)pydoc")
    .def(py::init<double, const string&, int, optional<uint32_t>>(),
	 R"pydoc(Constructor.

:param bandwidth: The bandwidth :math:`h>0`.
:type bandwidth: float
:param kernel: A string to select the kernel. Allowed values are ``'gaussian'``, 
    ``'exponential'``, and ``'laplacian'``. See :class:`NaiveKde` for 
    description.
:type kernel: str
:param random_samples: The number of random samples :math:`m>0`.
:type random_samples: int
:param seed: Optional pseudorandom number generator seed.
:type seed: int, optional
)pydoc", py::arg("bandwidth"), py::arg("kernel"),
	 py::arg("random_samples"), py::arg("seed") = nullopt);

  py::class_<RandomSamplingPermutedWrapper,KdeEstimatorWrapper>
    (m, "RandomSamplingPermuted", R"pydoc(The Permuted Random Sampling estimator.

Computes the unbiased estimate :math:`Z = \frac{1}{m}\sum_{x'\in X'} K_h(q,x')` 
where the set :math:`X'=\{x'_0,x'_1,\ldots,x'_{m-1}\}` is determined during 
preprocessing by permuting the dataset at random. During queries, a contiguous 
subset of points is used, and if Euclidean metric is used, the estimate is 
computed using matrix multiplication.)pydoc")
    .def(py::init<double, const string&, int, optional<uint32_t>>(),
	 R"pydoc(Constructor.

:param bandwidth: The bandwidth :math:`h>0`.
:type bandwidth: float
:param kernel: A string to select the kernel. Allowed values are ``'gaussian'``, 
    ``'exponential'``, and ``'laplacian'``. See :class:`NaiveKde` for 
    description.
:type kernel: str
:param random_samples: The number of random samples :math:`m>0`.
:type random_samples: int
:param seed: Optional pseudorandom number generator seed.
:type seed: int, optional)pydoc", py::arg("bandwidth"), py::arg("kernel"),
	 py::arg("random_samples"), py::arg("seed") = nullopt);

  py::class_<LinearScanWrapper>(m, "LinearScan",
				R"pydoc(A linear scan nearest neighbor algorithm.

This is a somewhat inefficient implementation of the linear scan that can be 
used for testing. Returns the exact :math:`k` nearest neighbors.)pydoc")
    .def(py::init<const string&>(), R"pydoc(Constructor.

:param metric: Sets the underlying metric, must be either ``'euclidean'`` or
    ``'taxicab'``.)pydoc",
	 py::arg("metric"))
    .def("fit", &LinearScanWrapper::fit, R"pydoc(Fits the dataset.

Basically just maintains a reference to the data matrix.

:param dataset: The dataset as an :math:`n\times d` matrix.
:type dataset: :class:`np.ndarray`)pydoc",
	 py::arg("dataset"))
    .def("query", &LinearScanWrapper::query, R"pydoc(Query for the :math:`k` nearest neighbors.

Returns the :math:`k` nearest neighbors for the input vectors, along with the 
distances and the number of samples looked at.

:param queries: Query vectors :math:`Q=\{q_0,q_1,\ldots,q_{m-1}\}` as an 
    :math:`m\times d` matrix, or an individual query vector of length 
    :math:`d`. The dimension :math:`d` must match that of the dataset.
:type queries: :class:`np.ndarray`
:param k_neighbors: The number of neighbors to query for.
:type k_neighbors: int
:return: A tuple :math:`(D,N,S)` of three arrays. Each 
    row in the arrays corresponds to a row in queries. The arrays have shapes 
    :math:`m\times k`, :math:`m\times k`, and :math:`m\times 1`, respectively. 
    The element :math:`D_{i,j}` records the distance from the query vector 
    :math:`q_i` to the :math:`j\textrm{th}` closest data vector.
    The element :math:`N_{i,j}` records the index of the the :math:`j\textrm{th}` closest
    data vector to the query vector :math:`q_i`. The element :math:`S_i` records
    how many data points were looked at during the query of the vector 
    :math:`q_i`.
:rtype: tuple
)pydoc",
	 py::arg("queries"), py::arg("k_neighbors"));

  

  py::class_<AnnEstimatorWrapper,KdeEstimatorWrapper>(m, "AnnEstimator",
						   R"pydoc(The DEANN estimator.

This computes a KDE estimate by taking in a conformant ANN object to query for 
approximate nearest neighbors and supplements the estimate with random samples.

The ANN object must provide a `query` function whose interface matches that of
:class:`LinearScan`.

This variant uses independent random sampling like :class:`RandomSampling`.
)pydoc")
    .def(py::init<double, const string&, int, int, py::object, optional<uint32_t>>(),
	 R"pydoc(Constructor.

:param bandwidth: The bandwidth :math:`h>0`.
:type bandwidth: float
:param kernel: A string to select the kernel. Allowed values are ``'gaussian'``, 
    ``'exponential'``, and ``'laplacian'``. See :class:`NaiveKde` for 
    description.
:type kernel: str
:param near_neighbors: The number of near neighbors to query :math:`k\geq 0`.
:type near_neighbors: int
:param random_samples: The number of random samples :math:`m\geq 0`.
:type random_samples: int
:param ann: The ANN object.
:type ann: object
:param seed: Optional pseudorandom number generator seed.
:type seed: int, optional
)pydoc", py::arg("bandwidth"), py::arg("kernel"),
	 py::arg("near_neighbors"), py::arg("random_samples"), py::arg("ann"),
	 py::arg("seed") = nullopt);

  

  py::class_<AnnEstimatorPermutedWrapper,KdeEstimatorWrapper>
    (m, "AnnEstimatorPermuted",
     R"pydoc(The DEANN estimator.

This computes a KDE estimate by taking in a conformant ANN object to query for 
approximate nearest neighbors and supplements the estimate with random samples.

The ANN object must provide a `query` function whose interface matches that of
:class:`LinearScan`.

This variant uses permuted random sampling like :class:`RandomSamplingPermuted`.)pydoc")
    .def(py::init<double, const string&, int, int, py::object, optional<uint32_t>>(),
	 R"pydoc(Constructor.

:param bandwidth: The bandwidth :math:`h>0`.
:type bandwidth: float
:param kernel: A string to select the kernel. Allowed values are ``'gaussian'``, 
    ``'exponential'``, and ``'laplacian'``. See :class:`NaiveKde` for 
    description.
:type kernel: str
:param near_neighbors: The number of near neighbors to query :math:`k\geq 0`.
:type near_neighbors: int
:param random_samples: The number of random samples :math:`m\geq 0`.
:type random_samples: int
:param ann: The ANN object.
:type ann: object
:param seed: Optional pseudorandom number generator seed.
:type seed: int, optional)pydoc", py::arg("bandwidth"), py::arg("kernel"),
	 py::arg("near_neighbors"), py::arg("random_samples"), py::arg("ann"),
	 py::arg("seed") = nullopt);

  vmlSetMode(VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_NOERR);
}
