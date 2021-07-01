#include <deann.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

// enable this define to print some timing information about different
// implementations
// #define DEANN_TESTS_DEBUG_PRINT_TIMES

using namespace deann;
using namespace Catch::literals;
using std::string;
using std::mt19937;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::cerr;
using std::endl;
using std::unique_ptr;
using std::make_unique;
using std::set;
using std::unordered_set;
using std::invalid_argument;
using std::nullopt;



namespace {
  class Allocator {
  public:
    template<typename T>
    T* allocate(size_t n) {
      size_t s = n*sizeof(T);
      ctr += s;
      T* ptr = new T[n];
      sizes.insert(std::make_pair(static_cast<void*>(ptr), s));
      return ptr;
    }

    template<typename T>
    void free(T* p) {
      auto it = sizes.find(static_cast<void*>(p));
      if (it == sizes.end()) {
	fprintf(stderr, "FATAL ERROR: TRIED TO FREE UNMANAGED POINTER %p\n", static_cast<void*>(p));
      }
      else {
	ctr -= it->second;
	sizes.erase(it);
      }
      delete[] p;
    }

    ~Allocator() {
      if (ctr > 0) {
	fprintf(stderr, "FATAL ERROR: MEMORY LEAK OR CORRUPTION OCCURRED: %lu BYTES UNFREED\n", ctr);
      }
    }

  private:
    size_t ctr = 0;
    std::map<void*,size_t> sizes;
  } allocator;
}


template<typename T>
static T relativeError(T wrong, T correct) {
  return std::abs((wrong-correct)/correct);
}



#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
static std::chrono::time_point<std::chrono::steady_clock> timerStart;



static void timerOn() {
  timerStart = std::chrono::steady_clock::now();
}



static std::chrono::nanoseconds	timerOff() {
  return std::chrono::steady_clock::now() - timerStart;
}



static string parseDuration(std::chrono::nanoseconds dur) {
  long hours = 0, minutes = 0;
  double seconds = 0;
  long nanos = dur.count();
  if (nanos > 3600l*1000000000l) {
    hours = nanos / (3600l*1000000000l);
    nanos -= hours*3600l*1000000000l;
  }
  if (nanos > 60l*1000000000l) {
    minutes = nanos / (60l*1000000000l);
    nanos -= minutes*60l*1000000000l;
  }
  if (nanos > 1000000000l) {
    seconds = static_cast<double>(nanos) / 1e9;
    nanos = 0;
  }

  char buffer[255];
  if (hours > 0) {
    sprintf(buffer, "%ld h %ld min %f s", hours, minutes, seconds);
  }
  else if (minutes > 0) {
    sprintf(buffer, "%ld min %f s", minutes, seconds);
  }
  else if (seconds > 0) {
    sprintf(buffer, "%f s", seconds);
  }
  else if (nanos > 1000000l) {
    double millis = static_cast<double>(nanos) / 1e6;
    sprintf(buffer, "%f ms", millis);
  }
  else if (nanos > 1000l) {
    double micros = static_cast<double>(nanos) / 1e3;
    sprintf(buffer, "%f µs", micros);
  }
  else {
    sprintf(buffer, "%ld ns", nanos);
  }
  return buffer;
}
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES



template<typename T>
static void constructMediumTestSet(T** X, int* np, int* dp) {
  mt19937 rng(0);
  const int n = 4347;
  const int d = 9;
  const int nc1 = 3738;
  const int nc2 = 609;
  static_assert(n == nc1+nc2);

  T mu1[d] = { 46.493792808219176, -0.4696329195205479, 85.95759310787672,
	       -0.08286065924657535, 41.49641481164384, 0.551664169520548,
	       39.45545269691781, 44.44643621575342, 5.158604452054795 };
  
  T A1[d*d] = { 10.484320588008893, 0.3933417436246518, 3.4489333938028346,
		0.07685890627519784, 8.341650647194841, 0.16124600516647916,
		-7.060789513966663, -4.841368513466814, 2.2086857803029716,
		0.0, 21.765355527204164, -0.2826628820265362,
		-0.08540511552349628, -0.03043131215905497, -1.4366990045589312,
		-0.2822081471726055, -0.24625189503899877, 0.03822928887457423,
		0.0, 0.0, 8.55993728342184,
		0.9514244361788197, -0.3364099775886661, -0.45018893824305206,
		8.52105984053665, 8.887347583067621, 0.3266303530113948,
		0.0, 0.0, 0.0,
		23.36493832376952, 0.11501621659842612, 3.0538978933372345,
		0.003091882814202935, -0.11986474667923924, -0.11576017116122887,
		0.0, 0.0, 0.0,
		0.0, 8.21593687925229, 0.623985315140864,
		-0.02795138746712225, -8.149163784066786, -8.007633004818983,
		0.0, 0.0, 0.0,
		0.0, 0.0, 39.30316959808004,
		-0.0010378511065134606, -0.003480475550055847, 0.03328552027023178,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.531827970590628, 0.1843911069690476, -0.2276178109314857,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.6705509315839024, 0.47000163064104533,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.653428935020192 };

  T mu2[d] = { 58.95957271980279, 1.8619556285949055, 81.56138044371406,
	       2.3857025472473294, -8.290879211175021, -0.46211996713229253,
	       22.529005751848807, 90.56433853738702, 68.00032867707478 };

  T A2[d*d] = { 16.070358405408477, 4.052170312280619, 1.8551219907830612,
		-3.6807561914034648, -6.398142299962961, 1.7564661882159314,
		-14.26236551530864, 8.424183992274793, 22.600158661730728,
		0.0, 51.620244224297, 0.08218075326467403,
		0.17943576700550545, 1.9184942987541551, 0.22811406617397098,
		0.09584620691959804, -1.8823707267045338, -1.9801929249019015,
		0.0, 0.0, 4.788239024721383,
		15.524965687634117, 3.503319037177967, -1.7898033254387289,
		4.7326906535881825, 1.170833428422221, -3.5439516919075045,
		0.0, 0.0, 0.0,
		91.6605723469673, -0.22001467927935622, 0.56208195978429,
		0.02047708618139976, 0.23974071441910022, 0.22749342416496315,
		0.0, 0.0, 0.0,
		0.0, 14.518184057649043, 3.719957972695156,
		0.004420457935305036, -14.899336759483328, -14.889768490545826,
		0.0, 0.0, 0.0,
		0.0, 0.0, 63.971143740807875,
		0.007063217124610587, 0.01864167735243769, 0.029553130574184044,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.5496925002026072, 0.19068829766895437, -0.17039726415544948,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.6996474217116678, 0.5758374884032743,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.6738895861785391 };
		
  *X = allocator.allocate<T>(n*d);
  *np = n;
  *dp = d;
  vector<int> indices(n);
  for (int i = 0; i < n; ++i)
    indices[i] = i;
  std::shuffle(indices.begin(), indices.end(), rng);
  
  normal_distribution<double> dist;
  T z[d];
  for (int j = 0; j < nc1; ++j) {
    for (int i = 0; i < d; ++i)
      z[i] = dist(rng);
    int idx = indices[j];
    T* x = *X + d*idx;
    array::matmulT(d,1,d,A1,z,x);
    array::add(d,x,mu1);
  }
  for (int j = nc1; j < n; ++j) {
    for (int i = 0; i < d; ++i)
      z[i] = dist(rng);
    int idx = indices[j];
    T* x = *X + d*idx;
    array::matmulT(d,1,d,A2,z,x);
    array::add(d,x,mu2);
  }
}



TEST_CASE( "test_sq_norm1", "[array]" ) {
  const int max_d = 100;
  mt19937 rng(0);
  normal_distribution<double> dist;
  float x[max_d];
  double y[max_d];

  for (int i = 0; i < max_d; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
  }

  REQUIRE(array::sqNorm(0, x) == 0);
  REQUIRE(array::sqNorm(0, y) == 0);

  for (int d : { 1, 2, 3, 5, 10, 20, 50, 100 } ) {
    float u = 0;
    for (float f : vector<float>(x, x+d))
      u += f*f;
    REQUIRE(std::abs(array::sqNorm(d,x) - u) < 1e-4);

    double v = 0;
    for (double f : vector<double>(y,y+d))
      v += f*f;
    REQUIRE(std::abs(array::sqNorm(d,y) - v) < 1e-13);
  }
}



TEST_CASE( "test_sq_norm2", "[array]" ) {
  const int n = 321;
  const int d = 1234;
  mt19937 rng(75432);
  normal_distribution<double> dist;
  float X[n*d];
  double Y[n*d];
  float V[n];
  double W[n];

  for (int i = 0; i < n*d; ++i) {
    X[i] = dist(rng);
    Y[i] = dist(rng);
  }

  array::sqNorm(n, 0, X, V);
  array::sqNorm(n, 0, Y, W);
  for (float f : V)
    REQUIRE(f == 0);
  for (double f : W)
    REQUIRE(f == 0);

  array::sqNorm(n, d, X, V);
  array::sqNorm(n, d, Y, W);
  for (int i = 0; i < n; ++i) {
    float u = 0;
    for (float f : vector<float>(X + i*d, X + (i+1)*d))
      u += f*f;
    REQUIRE(std::abs(V[i]-u) < 3e-3);

    double v = 0;
    for (double f : vector<double>(Y + i*d, Y + (i+1)*d))
      v += f*f;
    REQUIRE(std::abs(W[i] - v) < 4e-12);
  }
}



TEST_CASE( "test_ones", "[array]" ) {
  const int d = 1234;
  mt19937 rng(75432);
  normal_distribution<double> dist;
  float x1[d];
  float x2[d];
  double y1[d];
  double y2[d];
  for (int i = 0; i < d; ++i) {
    x1[i] = dist(rng);
    y1[i] = dist(rng);
  }
  array::mov(d,x1,x2);
  array::mov(d,y1,y2);
  REQUIRE(array::equal(d,x1,x2));
  REQUIRE(array::equal(d,y1,y2));

  for (int i = 0; i <= d; ++i) {
    array::ones(i, x2);
    array::ones(i, y2);
    for (int j = 0; j < i; ++j) {
      REQUIRE(x2[j] == 1);
      REQUIRE(y2[j] == 1);
    }
    REQUIRE(array::equal(d-i, x1+i, x2+i));
    REQUIRE(array::equal(d-i, y1+i, y2+i));
  }
}


  
TEST_CASE( "test_euclidean_distance1", "[array][euclidean_distance]" ) {
  const int max_d = 100;
  mt19937 rng(0);
  normal_distribution<double> dist;
  float x1[max_d];
  float x2[max_d];
  double y1[max_d];
  double y2[max_d];
  float scratch1[max_d];
  double scratch2[max_d];

  for (int i = 0; i < max_d; ++i) {
    x1[i] = dist(rng);
    x2[i] = dist(rng);
    y1[i] = dist(rng);
    y2[i] = dist(rng);
  }

  REQUIRE(array::euclideanDistance(0, x1, x2, scratch1) == 0);
  REQUIRE(array::euclideanDistance(0, y1, y2, scratch2) == 0);

  for (int d : { 1, 2, 3, 5, 10, 20, 50, 100 } ) {
    float u = 0;
    for (int i = 0; i < d; ++i) {
      u += (x1[i]-x2[i])*(x1[i]-x2[i]);
    }
    u = std::sqrt(u);
    REQUIRE(std::abs(array::euclideanDistance(d,x1,x2,scratch1) - u) < 1e-6);

    double v = 0;
    for (int i = 0; i < d; ++i) {
      v += (y1[i]-y2[i])*(y1[i]-y2[i]);
    }
    v = std::sqrt(v);
    REQUIRE(std::abs(array::euclideanDistance(d,y1,y2,scratch2) - v) < 2e-15);
  }
}



TEST_CASE( "test_euclidean_distance2", "[array][euclidean_distance]" ) {
  const int max_d = 1234;
  float x1[max_d];
  float y1[max_d];
  double x2[max_d];
  double y2[max_d];
  float scratch1[max_d];
  double scratch2[max_d];

  mt19937 rng(11223344);
  normal_distribution<double> dist;
  
  for (int i = 0; i < max_d; ++i) {
    x1[i] = dist(rng);
    y1[i] = dist(rng);
    x2[i] = dist(rng);
    y2[i] = dist(rng);
  }

  REQUIRE(array::euclideanDistance(0, x1, y1, scratch1) == 0);
  REQUIRE(array::euclideanDistance(0, x2, y2, scratch2) == 0);

  for (int d : { 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, max_d }) {
    float dist1 = 0;
    double dist2 = 0;
    for (int i = 0; i < d; ++i) {
      float ximyi1 = x1[i] - y1[i];
      dist1 += ximyi1*ximyi1;

      double ximyi2 = x2[i] - y2[i];
      dist2 += ximyi2*ximyi2;
    }
    dist1 = std::sqrt(dist1);
    dist2 = std::sqrt(dist2);
    REQUIRE(std::abs(dist1 - array::euclideanDistance(d, x1, y1, scratch1)) < 2e-5);
    REQUIRE(std::abs(dist2 - array::euclideanDistance(d, x2, y2, scratch2)) < 8e-15);
  }
}



TEST_CASE( "test_euclidean_distance3", "[array][euclidean_distance]" ) {
  mt19937 rng(11455);
  normal_distribution<double> nd;
  for (int n : { 1, 3, 5, 11, 101, 227 }) {
    for (int m : { 1, 2, 7, 13, 101, 229 }) {
      for (int d : { 0, 1, 2, 3, 5, 17, 101, 1381 }) {
	unique_ptr<float[]> X1 = make_unique<float[]>(n*d);
	unique_ptr<float[]> Q1 = make_unique<float[]>(m*d);
	unique_ptr<float[]> dists1 = make_unique<float[]>(m*n);	
	unique_ptr<float[]> scratch1 = make_unique<float[]>(std::max(m + std::max(m,n), d));
	unique_ptr<float[]> XSqNorms1 = make_unique<float[]>(n);

	for (int i = 0; i < n*d; ++i)
	  X1[i] = nd(rng);
	for (int i = 0; i < m*d; ++i)
	  Q1[i] = nd(rng);
	array::sqNorm(n, d, X1.get(), XSqNorms1.get());
	array::euclideanSqDistances(n, m, d, X1.get(), Q1.get(),
				    XSqNorms1.get(), dists1.get(),
				    scratch1.get());
	for (int i = 0; i < m; ++i) {
	  float* q = Q1.get() + i*d;
	  for (int j = 0; j < n; ++j) {
	    float* x = X1.get() + j*d;
	    float dist = array::euclideanDistance(d, x, q, scratch1.get());
	    float distSq = dist*dist;
	    float dist2Sq = dists1[i*n + j];
	    float absError = std::abs(dist2Sq - distSq);
	    REQUIRE(absError < 1e-3);
	  }
	}

	unique_ptr<double[]> X2 = make_unique<double[]>(n*d);
	unique_ptr<double[]> Q2 = make_unique<double[]>(m*d);
	unique_ptr<double[]> dists2 = make_unique<double[]>(m*n);	
	unique_ptr<double[]> scratch2 = make_unique<double[]>(std::max(m + std::max(m,n), d));
	unique_ptr<double[]> XSqNorms2 = make_unique<double[]>(n);
	for (int i = 0; i < n*d; ++i)
	  X2[i] = nd(rng);
	for (int i = 0; i < m*d; ++i)
	  Q2[i] = nd(rng);
	array::sqNorm(n, d, X2.get(), XSqNorms2.get());
	array::euclideanSqDistances(n, m, d, X2.get(), Q2.get(),
				    XSqNorms2.get(), dists2.get(),
				    scratch2.get());
	for (int i = 0; i < m; ++i) {
	  double* q = Q2.get() + i*d;
	  for (int j = 0; j < n; ++j) {
	    double* x = X2.get() + j*d;
	    double dist = array::euclideanDistance(d, x, q, scratch2.get());
	    double distSq = dist*dist;
	    double dist2Sq = dists2[i*n + j];
	    double relError = relativeError(dist2Sq, distSq);
	    double absError = std::abs(dist2Sq - distSq);
	    REQUIRE((dist == 0 || relError < 5e-8));
	    REQUIRE(absError < 2e-12);
	  }
	}
      }
    }
  }
}



TEST_CASE( "test_euclidean_distance4", "[array][euclidean_distance]" ) {
  mt19937 rng(0x7e5ccf43);
  normal_distribution nd;

  for (int n : { 1, 8, 101 }) {
    for (int m : { 1, 7, 103 }) {
      for (int d : { 0, 1, 11, 105 }) {
	vector<float> X(n*d);
	vector<float> Y(m*d);
	for (int i = 0; i < n*d; ++i)
	  X[i] = nd(rng); 
	for (int i = 0; i < m*d; ++i)
	  Y[i] = nd(rng);
	vector<float> Z1(m*n);
	vector<float> Z2(m*n);
	vector<float> XSqNorms(n);
	vector<float> scratch(std::max(d, m+std::max(m,n)));

	array::sqNorm(n, d, &X[0], &XSqNorms[0]);
	array::euclideanSqDistances(n, m, d, &X[0], &Y[0], &XSqNorms[0], &Z1[0],
				    &scratch[0]);

	for (int i = 0; i < m; ++i)
	  array::euclideanSqDistances(n, d, &X[0], &Y[0] + i*d, &XSqNorms[0],
				      &Z2[0] + i*n);

	const float DELTA = 1e-4f;
	for (int i = 0; i < m; ++i) {
	  float* y = &Y[0] + i*d;
	  for (int j = 0; j < n; ++j) {
	    float* x = &X[0] + j*d;
	    float mu = array::euclideanDistance(d, x, y, &scratch[0]);
	    if (d == 0) {
	      REQUIRE(mu == 0);
	    }
	    mu *= mu;
	    if (d > 0)
	      REQUIRE(std::abs(mu - Z1[i*n + j]) < DELTA);
	    else
	      REQUIRE(Z1[i*n + j] == 0);
	    if (d > 0)
	      REQUIRE(std::abs(mu - Z2[i*n + j]) < DELTA);
	    else
	      REQUIRE(Z1[i*n + j] == 0);
	    REQUIRE(std::abs(Z1[i*n + j] - Z2[i*n + j]) < DELTA);
	  }
	}
	REQUIRE(array::almostEqual(n*m, &Z1[0], &Z2[0], DELTA));
      }
    }
  }
}



TEST_CASE( "test_euclidean_distance5", "[array][euclidean_distance]" ) {
  mt19937 rng(0x6d5476c2);
  normal_distribution nd;

  for (int n : { 1, 8, 101 }) {
    for (int m : { 1, 7, 103 }) {
      for (int d : { 0, 1, 11, 105 }) {
	vector<double> X(n*d);
	vector<double> Y(m*d);
	for (int i = 0; i < n*d; ++i)
	  X[i] = nd(rng); 
	for (int i = 0; i < m*d; ++i)
	  Y[i] = nd(rng);
	vector<double> Z1(m*n);
	vector<double> Z2(m*n);
	vector<double> XSqNorms(n);
	vector<double> scratch(std::max(d, m+std::max(m,n)));

	array::sqNorm(n, d, &X[0], &XSqNorms[0]);
	array::euclideanSqDistances(n, m, d, &X[0], &Y[0], &XSqNorms[0], &Z1[0],
				    &scratch[0]);

	for (int i = 0; i < m; ++i)
	  array::euclideanSqDistances(n, d, &X[0], &Y[0] + i*d, &XSqNorms[0],
				      &Z2[0] + i*n);

	const double DELTA = 1e-12;
	for (int i = 0; i < m; ++i) {
	  double* y = &Y[0] + i*d;
	  for (int j = 0; j < n; ++j) {
	    double* x = &X[0] + j*d;
	    double mu = array::euclideanDistance(d, x, y, &scratch[0]);
	    if (d == 0) {
	      REQUIRE(mu == 0);
	    }
	    mu *= mu;
	    if (d > 0)
	      REQUIRE(std::abs(mu - Z1[i*n + j]) < DELTA);
	    else
	      REQUIRE(Z1[i*n + j] == 0);
	    if (d > 0)
	      REQUIRE(std::abs(mu - Z2[i*n + j]) < DELTA);
	    else
	      REQUIRE(Z1[i*n + j] == 0);
	    REQUIRE(std::abs(Z1[i*n + j] - Z2[i*n + j]) < DELTA);
	  }
	}
	REQUIRE(array::almostEqual(n*m, &Z1[0], &Z2[0], DELTA));
      }
    }
  }
}



TEST_CASE( "test_taxicab_distance1", "[array]" ) {
  const int max_d = 1234;
  float x1[max_d];
  float y1[max_d];
  double x2[max_d];
  double y2[max_d];
  float scratch1[max_d];
  double scratch2[max_d];

  mt19937 rng(12123434);
  normal_distribution<double> dist;
  
  for (int i = 0; i < max_d; ++i) {
    x1[i] = dist(rng);
    y1[i] = dist(rng);
    x2[i] = dist(rng);
    y2[i] = dist(rng);
  }

  REQUIRE(array::taxicabDistance(0, x1, y1, scratch1) == 0);
  REQUIRE(array::taxicabDistance(0, x2, y2, scratch2) == 0);

  for (int d : { 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, max_d }) {
    float dist1 = 0;
    double dist2 = 0;
    for (int i = 0; i < d; ++i) {
      dist1 += std::abs(x1[i] - y1[i]);
      dist2 += std::abs(x2[i] - y2[i]);
    }
    float re1 = relativeError(array::taxicabDistance(d, x1, y1, scratch1),dist1);
    REQUIRE(re1 < 6e-7);
    double re2 = relativeError(array::taxicabDistance(d, x2, y2, scratch2), dist2);
    REQUIRE(re2 < 3e-15);
  }
}



TEST_CASE( "test_taxicab_distance2", "[array]" ) {
  mt19937 rng(1414);
  normal_distribution<double> nd;
  const int max_d = 1231;
  float scratch1[max_d];
  double scratch2[max_d];

  for (int n : { 1, 3, 17, 101 }) {
    for (int m : { 1, 5, 23, 101 }) {
      for (int d : { 0, 1, 5, 13, 101, 1231 }) {
	vector<float> X1(n*d);
	vector<float> Q1(m*d);

	vector<double> X2(n*d);
	vector<double> Q2(m*d);

	vector<float> dists1(n*m);
	vector<double> dists2(n*m);
  
	for (int i = 0; i < n*d; ++i) {
	  X1[i] = nd(rng);
	  X2[i] = nd(rng);
	}
	for (int i = 0; i < m*d; ++i) {
	  Q1[i] = nd(rng);
	  Q2[i] = nd(rng);
	}

	array::taxicabDistances(n, m, d, &X1[0], &Q1[0], &dists1[0], &scratch1[0]);
	for (int i = 0; i < m; ++i) {
	  float* q = &Q1[0] + i*d;
	  for (int j = 0; j < n; ++j) {
	    float* x = &X1[0] + j*d;
	    if (d == 0) {
	      REQUIRE(dists1[i*n+j] == 0);
	    }
	    else {
	      REQUIRE(dists1[i*n+j] == array::taxicabDistance(d, q, x,
							      &scratch1[0]));
	    }
	  }
	}

	array::taxicabDistances(n, m, d, &X2[0], &Q2[0], &dists2[0],
				&scratch2[0]);
	for (int i = 0; i < m; ++i) {
	  double* q = &Q2[0] + i*d;
	  for (int j = 0; j < n; ++j) {
	    double* x = &X2[0] + j*d;
	    if (d == 0) {
	      REQUIRE(dists2[i*n+j] == 0);
	    }
	    else {
	      REQUIRE(dists2[i*n+j] == array::taxicabDistance(d, q, x,
							      &scratch2[0]));
	    }
	  }
	}
      }
    }
  }
}




TEST_CASE( "test_add1", "[array]" ) {
  const int max_d = 100;
  mt19937 rng(0);
  normal_distribution<double> dist;
  float x1[max_d];
  float x2[max_d];
  float x3[max_d];
  float x4[max_d];
  double y1[max_d];
  double y2[max_d];
  double y3[max_d];
  double y4[max_d];

  for (int i = 0; i < max_d; ++i) {
    x3[i] = x1[i] = dist(rng);
    x4[i] = x2[i] = dist(rng);
    y3[i] = y1[i] = dist(rng);
    y4[i] = y2[i] = dist(rng);
  }

  array::add(0, x3, x4);
  array::add(0, y3, y4);
  for (int i = 0; i < max_d; ++i) {
    REQUIRE(x1[i] == x3[i]);
    REQUIRE(x2[i] == x4[i]);
    REQUIRE(y1[i] == y3[i]);
    REQUIRE(y2[i] == y4[i]);
  }

  for (int d : { 1, 2, 3, 5, 10, 20, 50, 100 } ) {
    for (int i = 0; i < max_d; ++i) {
      x3[i] = x1[i];
      x4[i] = x2[i];
      y3[i] = y1[i];
      y4[i] = y2[i];
    }
    
    array::add(d,x3,x4);
    array::add(d,y3,y4);

    for (int i = 0; i < d; ++i) {
      REQUIRE(x3[i] == x1[i] + x2[i]);
      REQUIRE(x4[i] == x2[i]);
      REQUIRE(y3[i] == y1[i] + y2[i]);
      REQUIRE(y4[i] == y2[i]);
    }
    for (int i = d; i < max_d; ++i) {
      REQUIRE(x3[i] == x1[i]);
      REQUIRE(x4[i] == x2[i]);
      REQUIRE(y3[i] == y1[i]);
      REQUIRE(y4[i] == y2[i]);
    }
  }
}



TEST_CASE( "test_add2", "[array]" ) {
  const int d = 100;
  mt19937 rng(112233);
  normal_distribution<double> dist;
  float x1[d];
  float x2[d];
  
  for (int j = 0; j < d; ++j) {
    x1[j] = dist(rng);
  }

  array::mov(d, x1, x2);
  REQUIRE(array::equal(d, x1,x2));
  
  array::add(d, x2, 0.0f);
  REQUIRE(array::equal(d, x1, x2));
  
  const int nRepeats = 100;
  for (int i = 0; i < nRepeats; ++i) {
    array::mov(d, x1, x2);
    REQUIRE(array::equal(d, x1, x2));
    float c = dist(rng);
    array::add(d, x2, c);
    for (int j = 0; j < d; ++j)
      REQUIRE(x1[j] + c == x2[j]);
  }

  double y1[d];
  double y2[d];
  
  for (int j = 0; j < d; ++j) {
    y1[j] = dist(rng);
  }

  array::mov(d, y1, y2);
  REQUIRE(array::equal(d, y1,y2));
  
  array::add(d, y2, 0.0);
  REQUIRE(array::equal(d, y1, y2));
  
  for (int i = 0; i < nRepeats; ++i) {
    array::mov(d, y1, y2);
    REQUIRE(array::equal(d, y1, y2));
    double c = dist(rng);
    array::add(d, y2, c);
    for (int j = 0; j < d; ++j)
      REQUIRE(y1[j] + c == y2[j]);
  }       
}



TEST_CASE( "test_sum_mean", "[array]" ) {
  const int d = 1234;
  mt19937 rng(112233);
  normal_distribution<double> dist;
  float x[d];
  double y[d];
  float S1 = 0.0;
  double S2 = 0.0;
  float m1 = 0.0;
  double m2 = 0.0;
  for (int i = 0; i < d; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
  }
  for (int i = 0; i < d; ++i) {
    S1 += x[i];
    S2 += y[i];
  }
  m1 = S1/d;
  m2 = S2/d;
  float T1 = array::sum(d,x);
  double T2 = array::sum(d,y);
  REQUIRE(T1 == S1);
  REQUIRE(T2 == S2);
  float m3 = array::mean(d,x);
  double m4 = array::mean(d,y);
  REQUIRE(m1 == m3);
  REQUIRE(m2 == m4);
}


TEST_CASE( "test_mul", "[array]" ) {
  const int max_d = 100;
  mt19937 rng(0);
  normal_distribution<double> dist;
  float x1[max_d];
  float x2[max_d];
  for (int i = 0; i < max_d; ++i)
    x1[i] = dist(rng);

  int n_reps = 100;
  for (int i = 0; i < n_reps; ++i) {
    float c = dist(rng);
    array::mov(max_d, x1, x2);
    REQUIRE(array::equal(max_d, x1, x2));
    array::mul(0, x2, c);
    REQUIRE(array::equal(max_d, x1, x2));
    array::mul(1, x2, c);
    REQUIRE(x2[0] == c*x1[0]);
    REQUIRE(array::equal(max_d-1, x1+1, x2+1));

    for (int d : { 2, 3, 5, 10, 20, 50, 100 }) {
      array::mov(max_d, x1, x2);
      REQUIRE(array::equal(max_d, x1, x2));
      array::mul(d, x2, c);
      for (int j = 0; j < d; ++j)
	REQUIRE(x2[j] == c*x1[j]);
      REQUIRE(array::equal(max_d-d, x1+d, x2+d));
    }
  }

  double x3[max_d];
  double x4[max_d];
  for (int i = 0; i < max_d; ++i)
    x3[i] = dist(rng);

  for (int i = 0; i < n_reps; ++i) {
    double c = dist(rng);
    array::mov(max_d, x3, x4);
    REQUIRE(array::equal(max_d, x3, x4));
    array::mul(0, x4, c);
    REQUIRE(array::equal(max_d, x3, x4));
    array::mul(1, x4, c);
    REQUIRE(x4[0] == c*x3[0]);
    REQUIRE(array::equal(max_d-1, x3+1, x4+1));

    for (int d : { 2, 3, 5, 10, 20, 50, 100 }) {
      array::mov(max_d, x3, x4); 
      REQUIRE(array::equal(max_d, x3, x4));
      array::mul(d, x4, c);
      for (int j = 0; j < d; ++j)
	REQUIRE(x4[j] == c*x3[j]); 
      REQUIRE(array::equal(max_d-d, x3+d, x4+d));
    }
  }
}



TEST_CASE( "test_rowwise_sum", "[array]" ) {
  const int n = 1234;
  const int d = 100;
  mt19937 rng(0);
  normal_distribution<double> dist;
  float x1[n*d];
  float y1[n];
  for (int i = 0; i < n*d; ++i)
    x1[i] = dist(rng);
  array::rowwiseSum(n,d,x1,y1);
  for (int i = 0; i < n; ++i) {
    float c = 0.0;
    for (int j = 0; j < d; ++j)
      c += x1[i*d+j];
    REQUIRE(c == y1[i]);
  }  

  double x2[n*d];
  double y2[n];
  for (int i = 0; i < n*d; ++i)
    x2[i] = dist(rng);
  array::rowwiseSum(n,d,x2,y2);
  for (int i = 0; i < n; ++i) {
    double c = 0.0;
    for (int j = 0; j < d; ++j)
      c += x2[i*d+j];
    REQUIRE(c == y2[i]);
  }  
}



TEST_CASE( "test_unary", "[array]" ) {
  const int n = 1234;
  mt19937 rng(31415);
  normal_distribution<double> dist;
  float x1[n];
  float y1[n];
  for (int i = 0; i < n; ++i)
    x1[i] = dist(rng);
  array::mov(n,x1,y1);
  REQUIRE(array::equal(n,x1,y1));

  const float h1 = 16;

  array::unary(n, y1, [&](float x) {
      return std::exp(-std::sqrt(std::abs(x))/h1);
    });
  for (int i = 0; i < n; ++i) {
    REQUIRE(y1[i] == std::exp(-std::sqrt(std::abs(x1[i]))/h1));
  }

  double x2[n];
  double y2[n];
  for (int i = 0; i < n; ++i)
    x2[i] = dist(rng);
  array::mov(n,x2,y2);
  REQUIRE(array::equal(n,x2,y2));

  const double h2 = 17;

  array::unary(n, y2, [&](double x) {
      return std::exp(-std::sqrt(std::abs(x))/h2);
    });
  for (int i = 0; i < n; ++i) {
    REQUIRE(y2[i] == std::exp(-std::sqrt(std::abs(x2[i]))/h2));
  } 
}



TEST_CASE( "test_mean", "[array]" ) {
  const int n = 1234;
  const int nRepeats = 1000;
  mt19937 rng(212121);
  normal_distribution<double> dist;
  float x1[n];
  for (int k = 0; k < nRepeats; ++k) {
    for (int i = 0; i < n; ++i)
      x1[i] = dist(rng);
    float m = 0;
    for (int i = 0; i < n; ++i)
      m += x1[i];
    m /= n;
    REQUIRE(m == array::mean(n, x1));
  }


  double x2[n];
  for (int k = 0; k < nRepeats; ++k) {
    for (int i = 0; i < n; ++i)
      x2[i] = dist(rng);
    double m = 0;
    for (int i = 0; i < n; ++i)
      m += x2[i];
    m /= n;
    REQUIRE(m == array::mean(n, x2));
  }
}



TEST_CASE( "test_matmul1", "[array][matmul]" ) {
  const int n1 = 13;
  const int d1 = 11;
  const int m1 = 17;

  float A1[n1*d1] = { -1.156074286f, 0.059535675f, 0.376610368f, -0.546119153f,
		      0.605356097f, 0.977344096f, 0.407545805f, 0.557708621f,
		      0.300337344f, 0.214505181f, 0.183614388f, 0.183265328f,
		      0.504055321f, -0.256151259f, -1.162897468f, 2.162698030f,
		      0.741736889f, 0.020884419f, -1.191972733f, 0.397885442f,
		      0.072022736f, 0.300132066f, -0.912516594f, 0.627277136f,
		      -0.670268476f, 0.489361227f, -0.935610652f, -1.780584931f,
		      -0.090447523f, 0.173366696f, -0.905955195f, 0.383688390f,
		      -0.784931719f, -0.418320745f, -0.354602545f, 1.568362832f,
		      -1.288706064f, -0.569747388f, -0.974142015f,
		      -0.459173590f, -0.056367893f, 2.114112854f, 1.139562011f,
		      -1.946816802f, 0.813374639f, 1.555392742f, 0.213038921f,
		      -1.929827332f, -0.231557935f, -1.027695894f,
		      -0.032223582f, -0.162669644f, -0.206813812f, 0.319757998f,
		      -1.191838741f, -0.205754355f, -1.243185401f, 1.009676337f,
		      -1.577310681f, -0.083870150f, 0.793422639f, -1.368369460f,
		      0.055673640f, -0.184471726f, -1.544784307f, 0.980010450f,
		      0.950200200f, -2.185411215f, 0.668090940f, 1.711692810f,
		      0.643442273f, 0.393582821f, 0.201390609f, 0.254097044f,
		      0.324999005f, 0.021879951f, 0.029451638f, 0.902588665f,
		      -0.061802518f, 1.286408305f, 1.164627910f, -0.044181883f,
		      -1.286036730f, 1.468623400f, -0.573981106f, -1.846693873f,
		      -0.373945206f, -0.170846179f, -1.101605296f,
		      -1.009706020f, -0.634907663f, -2.172125340f, 0.490949035f,
		      -0.869945109f, 1.598746061f, 0.215125456f, 0.464810461f,
		      1.763559699f, -1.462650299f, -0.704895973f, -0.386487603f,
		      0.222564727f, -0.697512984f, 0.076289929f, -0.368060350f,
		      -0.536768794f, -0.283323139f, 1.448760152f, -0.223966792f,
		      1.115548491f, -0.338850588f, -0.625648022f, 0.831737161f,
		      0.573971987f, -0.553234458f, -0.205810577f, 1.450484276f,
		      1.469928503f, 0.335611492f, 0.838402033f, 1.107342124f,
		      -0.748771846f, -0.224420458f, -1.009743690f,
		      -0.821726799f, 0.296085060f, -0.690578938f, 0.781220019f,
		      0.960092425f, -0.534077346f, -1.031394601f, 1.493485928f,
		      0.018029554f, -0.613610029f, -1.283669353f, -1.055390358f,
		      -0.670329213f, 0.028152963f, 2.449367523f, -0.610049546f,
		      -0.789321601f, -0.048152301f, -0.532641470f };

  float B1[m1*d1] = { 1.690127611f, 0.231585175f, -1.540844083f, -1.521862030f,
		      -0.659878492f, 0.825400651f, -0.774252415f, -1.554029822f,
		      1.165902734f, 0.996075273f, -0.268112063f, -0.578488469f,
		      0.113871649f, 0.391512036f, 0.888956249f, -0.012769731f,
		      -1.572849989f, -2.194645643f, 1.102235794f, -0.379043370f,
		      0.513080001f, 0.191210479f, -1.373879790f, -0.853624761f,
		      -0.542115211f, -1.011974216f, -0.214377373f, 0.090042695f,
		      0.811273694f, -0.313428640f, 0.388074726f, 0.393736690f,
		      2.220658302f, 0.824833691f, 0.151596919f, -1.239938498f,
		      0.300651699f, -0.607140481f, -0.823965728f, -0.876483440f,
		      -0.301605999f, 0.304357886f, 0.620918930f, -0.589303195f,
		      -0.342500210f, -0.345265299f, -0.195206359f, 1.325548291f,
		      -0.428822726f, 0.038367987f, -0.245632306f, 0.199577332f,
		      -1.592611194f, -1.342900753f, -0.572182596f,
		      -0.996570170f, 1.895925641f, -0.091853939f, -0.503936172f,
		      0.326266259f, 0.213462010f, -0.275385410f, -0.870673239f,
		      -1.183350801f, 0.507957518f, -1.137328625f, 2.831827879f,
		      0.396805495f, 2.082157135f, 0.426801652f, 0.436578333f,
		      -0.429902107f, -0.860102713f, 2.140659094f, -1.345444918f,
		      1.351800680f, 0.717468679f, 0.018854246f, 0.665760159f,
		      -0.118093960f, -1.173003912f, 1.295635819f, -1.031835318f,
		      -1.450793505f, 0.418104947f, 0.887893736f, 0.620804965f,
		      -0.455685616f, -0.094898641f, 1.768086076f, -0.229509756f,
		      -0.245796829f, -2.261720657f, 0.875320137f, -0.914619088f,
		      0.534037888f, 0.320951939f, 2.140745163f, -0.067477897f,
		      0.161802828f, 2.506873131f, 0.375408947f, 0.689160466f,
		      0.644354463f, -0.667655826f, -0.649420202f, 0.985804319f,
		      -0.630884826f, 0.615130246f, 0.749806285f, 0.298884749f,
		      -1.036102414f, -1.367203355f, -1.276271105f, 0.474622577f,
		      -0.970868468f, -0.855388343f, 1.314132452f, 0.169496492f,
		      -0.386474103f, -0.834428191f, -0.361174107f, 0.178063110f,
		      0.019948903f, -2.152659655f, -0.626566648f, 0.720753729f,
		      -1.838777065f, 0.440858126f, -0.030343782f, 0.745530188f,
		      -0.200162530f, -0.233621672f, 0.063930452f, -0.216862500f,
		      1.559813976f, 0.699358582f, 1.217607498f, -2.174288273f,
		      -0.659002244f, -0.128868505f, -1.058634162f,
		      -0.309751451f, -0.286437809f, 0.185387760f, 1.544280291f,
		      4.498898506f, 0.251646072f, -0.146594107f, -0.103624538f,
		      -0.885110438f, 0.039919961f, 0.051747076f, -1.162895799f,
		      0.498730212f, -1.881626368f, 0.209951118f, -0.609310508f,
		      -0.533538759f, 0.524455965f, 2.360199213f, -1.847912908f,
		      -0.844411790f, -0.961292148f, -3.087707043f, 0.935638547f,
		      -1.226976514f, -1.804933071f, 0.971972525f, 0.778137207f,
		      -0.246892482f, -1.502307534f, -1.531836748f, 0.839057446f,
		      1.278603435f, 0.081315242f, -1.187228680f, -0.416669697f,
		      -0.955183566f, -0.088225186f, -0.239456788f, 0.521443665f,
		      -1.189121723f, 1.108298302f, -0.096562997f, 0.037718017f,
		      -0.972552121f };

  float C1[n1*m1] = { -1.949707150f, -1.455785275f, 2.608792543f, -3.157562733f,
		      -1.504356861f, 0.858644485f, -1.994027257f, 0.347905993f,
		      0.217408538f, -0.169985443f, -0.687596321f, 1.546933889f,
		      -0.622559011f, -2.250399828f, -1.182639837f,
		      -3.060914278f, 1.334841132f, 4.067402363f, -3.793071270f,
		      1.476580977f, -1.398513556f, -3.772377968f, 2.502981901f,
		      -2.477580309f, 3.502644777f, -3.417742014f, 0.123699971f,
		      -0.121865943f, 1.118964553f, 1.166318417f, -4.427423954f,
		      -0.047113050f, 2.629659414f, -1.884926915f, -2.624279976f,
		      4.364215374f, -1.444671273f, 2.807880640f, 2.641886473f,
		      3.261919737f, -1.541584969f, 0.525795043f, 2.545620441f,
		      2.545102119f, 1.312148690f, -0.609269500f, -1.624319434f,
		      2.322160244f, -0.222990379f, -0.689504564f, 1.879653335f,
		      2.892418861f, 1.566185832f, -2.043191433f, 1.335396528f,
		      -5.223550797f, 0.322929651f, -0.866187990f, 5.463732719f,
		      3.444732904f, -2.822593927f, 2.279290199f, 4.542821407f,
		      -3.700052738f, -0.817470551f, 3.517177582f, -0.393747360f,
		      1.104244709f, 4.322790146f, -0.400189161f, -3.226834774f,
		      1.964963317f, -2.798033237f, 4.709517002f, 2.416061878f,
		      4.586507797f, 3.363654137f, 2.611822605f, 2.340795040f,
		      4.016833782f, -4.325646400f, -6.663758278f, 2.055991650f,
		      -2.898126125f, -1.078227997f, -0.124328598f, 0.252904087f,
		      2.851023436f, -3.097458601f, 0.432976156f, -2.660444021f,
		      0.134631887f, 0.411186755f, -3.289432287f, -3.597069263f,
		      2.040751457f, 5.091338634f, 2.490856171f, -6.980322838f,
		      -0.847218215f, -2.363636494f, 1.066112399f, -2.792250395f,
		      0.089148849f, -1.352928758f, -0.734278321f, 1.744209766f,
		      -6.403985977f, 4.041113377f, -2.993419647f, -5.538342476f,
		      -3.792330027f, -0.442709953f, -4.780912399f, 2.413064003f,
		      7.893435478f, 3.165819168f, 4.043141842f, -0.972582817f,
		      -6.000559330f, -0.347244918f, -3.041497946f, -1.231001735f,
		      4.040157318f, 0.274298459f, 5.605286598f, -4.437039852f,
		      -4.833649635f, 1.394153833f, -3.720216274f, -6.859684944f,
		      -2.402157068f, 7.594988346f, 6.449277878f, -3.355015278f,
		      -5.089488983f, 2.264834166f, -3.117757797f, 3.590437174f,
		      0.123497009f, -4.894724846f, 1.692628860f, -5.386305809f,
		      3.408161640f, -0.197105169f, -4.678885460f, 4.967636585f,
		      2.695185184f, -8.865226746f, -8.918572426f, 7.626771450f,
		      -0.365675449f, 1.736930013f, 1.106393337f, 0.823749125f,
		      4.438701630f, -0.668335915f, -3.209542751f, -2.455293179f,
		      -3.388689518f, 2.299173117f, -0.740915716f, -1.331216931f,
		      0.657323420f, 1.806504130f, -0.332427531f, -3.535552502f,
		      -4.582197189f, 1.145576954f, -0.272328794f, -5.155327320f,
		      0.243022785f, 3.703337908f, -2.472106457f, -1.197031498f,
		      -4.348526955f, 3.991240263f, -2.874739408f, 0.951252341f,
		      0.506859243f, -1.885681987f, -2.635560513f, -5.279692650f,
		      1.056404829f, -2.780988216f, -3.859549761f, -1.330377579f,
		      -3.423788786f, -0.333241284f, 5.509626865f, -1.914794207f,
		      0.669156075f, -1.864795208f, -2.457295418f, -0.511240661f,
		      -3.848089457f, 0.284185112f, 2.904305935f, -0.768918157f,
		      -3.169182539f, -7.649779320f, -3.281947374f, -3.290264368f,
		      0.283742726f, 2.164214134f, -7.432068348f, 3.079410076f,
		      -0.339080006f, 0.248943895f, 0.627743661f, -6.418297768f,
		      -4.212759018f, -1.878362298f, -5.318370819f, 0.830093741f,
		      -2.107306480f, -6.451720715f, -6.150620461f, 11.175296783f,
		      -1.980050802f, -1.267574191f };

  float A2[n1*d1];
  float B2[m1*d1];
  float C2[n1*m1];
  array::mov(n1*d1, A1, A2);
  array::mov(m1*d1, B1, B2);
  array::matmulT(n1, m1, d1, A2, B2, C2);

  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < d1; ++j) {
      int idx = i*d1 + j;
      REQUIRE(A1[idx] == A2[idx]);
    }
  }
  REQUIRE(array::equal(n1*d1,A1,A2));

  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < d1; ++j) {
      int idx = i*d1 + j;
      REQUIRE(B1[idx] == B2[idx]);
    }
  }
  REQUIRE(array::equal(m1*d1,B1,B2));

  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < m1; ++j) {
      int idx = i*m1 + j;
      REQUIRE(std::abs(C1[idx] - C2[idx]) < 1e-6);
    }
  }
  REQUIRE(!array::equal(n1*m1,C1,C2));
  REQUIRE(array::almostEqual(n1*m1,C1,C2,1e-6f));

  const int n2 = 17;
  const int m2 = 19;
  const int d2 = 13;
  
  double A3[n2*d2] = { -1.66148113478158299, -0.95880806041796407,
		       -0.77565753483951005, -0.04118031193690145,
		       -1.27401967091746959, 1.63840974026653274,
		       -2.12525560410730829, 0.31120746112731912,
		       0.78022754279998874, -0.31589517819068147,
		       0.12973649107567070, -0.09841288842948762,
		       -0.33249902104065898, -0.09994735062206529,
		       0.39316318794905941, 1.87809424079299125,
		       -0.45938989924639539, 0.93154671609719530,
		       -0.53783907674827525, -0.71559832537783807,
		       0.83360672373055644, -0.47298728637185000,
		       0.35840536294434394, 0.99897778599060638,
		       0.08378395025710648, -0.88715732329634267,
		       -0.16055497719505810, 0.74018861872309927,
		       0.11502701252621375, -0.34202828076158392,
		       0.61950333988523298, -1.89540546575795843,
		       -0.19247276939906238, 1.81854798113588223,
		       -0.07582265822321722, 0.04079536593533470,
		       -0.95072698650025256, 0.71566542287866708,
		       -0.31509970404257281, -1.25487352178258371,
		       1.07412282429194961, 1.26375151368083904,
		       -1.27446623399867054, -0.99671153390096479,
		       0.91740450862502165, -1.06844562860669012,
		       0.95698842125844319, 0.16131671594220037,
		       0.64744515043887518, 1.63811836951215284,
		       0.06835678369920563, 0.21893285948663463,
		       1.20546593259555390, 1.69861936113545164,
		       -0.29037571562199682, 1.06357151496410207,
		       -0.26095198003120929, 0.21461536512251081,
		       1.10175594941166710, 0.19979095557774934,
		       1.45455326237447191, 1.47272836547888963,
		       0.50741765097262770, -0.99012994161990153,
		       -1.21818313655366728, 0.27702289721456791,
		       -0.16960316941453105, 0.27101218258603449,
		       0.00398965704956090, 1.26488229135418884,
		       0.57366533682993825, 0.43075681238542268,
		       0.25262761775056564, -1.01042129936282676,
		       1.44973709119154526, 1.78868850008631530,
		       0.63734819273105647, -2.20571806007118987,
		       -1.04607824162129326, -0.75711030360922937,
		       -0.76633479879007393, -0.22193533490349107,
		       -1.05035673717978306, 0.22557443146692660,
		       2.27493734174345574, 0.91310170420665659,
		       -1.37694207219477982, -0.53177814786674316,
		       -0.03957341760707329, 1.08045134914997387,
		       -0.78515821065984670, -0.51196498275975189,
		       0.73025800965759935, 0.43243969281785238,
		       1.33365801154062047, -0.95162605729835492,
		       -0.12652130627266164, -0.57966269785536984,
		       2.14631921676543502, -1.21581293744858687,
		       -0.23512093557530514, 1.03177827709661707,
		       -1.03671558326384594, -2.10563328212422052,
		       -0.13290929543340216, 0.74349341154677184,
		       -0.35059564762824619, -1.62555146414371765,
		       -0.06832493415901113, 1.45452909753087534,
		       -0.09613399880783945, -0.19602598239903268,
		       1.48400656499950601, 0.08206117081278179,
		       -0.96551524598871696, 0.46160383630660745,
		       -0.07759786304371670, 1.02685561063965158,
		       -0.22756949719838857, 1.71884532352276231,
		       0.48873804547139527, -0.15178166357103556,
		       0.43955281825691800, -0.64224181366783795,
		       0.55428548401780908, 1.12024959908012223,
		       -0.46844482660473141, 1.68266779789987808,
		       0.36316816686244535, 0.18974701728403645,
		       2.22272932459008388, 0.04452426760823655,
		       0.64621771036749320, -0.49432567662512356,
		       -2.25396300306460562, -3.31940535726488406,
		       1.02506703779203989, -0.13407013236542437,
		       -0.44713893820592543, -0.81178908331958377,
		       0.51975526803364980, -0.35131308690954288,
		       -2.51870112547448866, -0.10659948195659699,
		       -1.57935660029507363, -1.82666421225533049,
		       -0.47714068573256729, 0.73125234134124439,
		       1.16060107557060888, -0.02872530166979339,
		       -0.27597963401784337, 0.89462012235458332,
		       -0.11050539543689977, -0.31188648695702037,
		       -0.88792801855765524, 0.59444074164108396,
		       0.33066422459392575, 1.40231116730225391,
		       0.76567388577854023, 0.90654147667783302,
		       0.12753933267552647, 0.89071071940123814,
		       0.13119247475209495, -0.41961522615130759,
		       -2.22689383367471372, 1.16222111960258778,
		       -1.29042891664981396, 0.69737532476849218,
		       -0.51324991973611267, 1.30777057528213536,
		       0.12189480885807838, -0.23759176561026848,
		       -0.94439816754678352, -2.48338350685521858,
		       -1.11942734614813610, 0.96211283033516537,
		       -0.88077224964516121, -0.03078141305685048,
		       0.19731857202583367, -0.79043862314259183,
		       0.88853859183544426, -0.33811556494345579,
		       -1.19872051150103665, 0.78660986134291688,
		       1.50184108611737832, -0.64408142293916237,
		       -2.85646582926659542, -0.43078422956510709,
		       -2.36395112242353944, 0.58313138437548506,
		       1.51974189915361624, -0.00350035703734618,
		       0.43001221378787535, -0.81444130649127600,
		       -0.45139129033163089, 0.88206937509572514,
		       -0.16073931761213425, 0.71519388851581511,
		       -2.05355613679119120, 0.98658336731763996,
		       0.99601164098960693, -0.15394106405634450,
		       -2.23963104644241495, 0.84092779281810659,
		       0.20682490582182603, -1.24556396652701773,
		       2.24480693945601040, -0.40307972084503552,
		       -0.72687593989374877, -0.51078295289224429,
		       1.15759455820308732, -1.87786255832894455,
		       0.10959227136695754, 0.31908921707447291,
		       -0.38090372393130900, 2.25646933799432725,
		       -0.72554454670725488, 1.11737006309950759,
		       1.69252446594710548, 0.41145210115175074,
		       -0.89627574134838317 };
  double B3[m2*d2] = { 0.59184911062636703, 0.40108806429074628,
		       -2.22366467465052065, -0.59750856551188269,
		       -0.23783449488903677, 1.41172166356854523,
		       1.37584744825998784, -0.52589235851693428,
		       -1.13139513629703115, 0.46218985360892678,
		       -1.64128348261504198, -1.06985194689018703,
		       1.05885733948820215, 0.36415721747767105,
		       1.09422572701234455, 0.39386339451695662,
		       0.33984495151589122, 0.49497848345884488,
		       0.42479846957554573, -1.44949235340853222,
		       0.34513905738597755, -0.38274732433192121,
		       0.80260416012018554, 1.05177834970481587,
		       -0.10833157000745566, -0.86440750770948238,
		       0.45296674086279920, -0.34880102847941791,
		       -0.31469508706932264, 0.76965614709260188,
		       -0.99671064700218726, -0.16654179481054568,
		       1.09799505226890215, 0.06133690273189082,
		       0.62355634223152723, -0.15756496960846317,
		       1.01336193609304948, 0.58495519944675889,
		       -2.02854229767011063, -0.30199951988345630,
		       0.26874234302018885, -0.74905702206548597,
		       1.50475252474218779, -0.02545611029327325,
		       0.38396057451964272, 0.06195813784130217,
		       0.26549852034588473, 0.88461461730525992,
		       0.44325215987139072, -0.89796410260601711,
		       -1.20816745987462104, 0.04233554726974770,
		       -0.70486410219891427, 0.00002996999659865,
		       -1.13533837668220383, -0.56014780202598236,
		       -0.22192820361804244, 0.14413098403078708,
		       -0.35797097657395660, -2.24243459417333391,
		       -0.03769880089090479, 0.95259977498276927,
		       -0.26553950688032762, -1.08396893287965557,
		       -1.11152915030601362, 0.77014927586047266,
		       -1.78535025330442765, -0.55992253329439445,
		       -0.21377555069321685, -0.36461287514627416,
		       1.55094246711988837, 0.01666209823887382,
		       0.02912155156829500, 0.30732995490098458,
		       0.60316086544359804, 1.00074386483294875,
		       0.20612895618306221, 1.32861466544944129,
		       -0.08989191716428879, 1.00053028985944725,
		       0.00495808948562157, -0.36079502407395642,
		       1.18827400003186345, 0.07464763314272289,
		       0.10319095215358826, -0.35134430671004568,
		       -1.89747929148447714, 0.02994629534388396,
		       1.78171053302175464, -1.52685951154650179,
		       0.48444082507303426, -0.25445138146686508,
		       -1.73117467096855027, -0.62875972741437280,
		       0.01828081376221203, 0.59446553488461373,
		       -0.00627288468831754, -1.31343545160690667,
		       -0.26242949416731232, 0.09283229671282509,
		       -1.46090018940002064, -0.85309944120457848,
		       -0.50402095349618825, -2.27107914136572919,
		       -1.41965629828960838, 1.09304117743935092,
		       -1.22076955038095480, -0.79876998103167463,
		       0.48229035757833727, 1.97101379493898698,
		       0.38216181864513366, -1.32512469469414507,
		       -0.00028748668951756, 1.28489065252492685,
		       -0.71840102340162437, -1.60940395277727544,
		       0.92952588322015484, -0.30454508207826819,
		       0.15833088710973164, 0.23054802056732240,
		       -2.08513181108146473, -0.06377181181657507,
		       1.33166522985132607, 0.78246627681756353,
		       -0.68958011872116409, 0.62901505602140817,
		       -0.71354974939491922, 0.84816597385030357,
		       -0.72126671396567055, -0.80513694453894658,
		       -1.12773783253620241, -1.75262473391211837,
		       0.01421818111965697, 0.28139828511710901,
		       1.47654778857070856, -0.96415581897372582,
		       1.35650001551905719, 1.20194108931288968,
		       0.02418917594741118, -0.92397352002226529,
		       1.20627074497095799, -0.43815989356167567,
		       -0.98190094387803162, -1.00084926405850072,
		       -1.36221307528393543, -1.10475801470378654,
		       0.86394573259617491, -1.61453268408243678,
		       0.47169727026134589, -0.35027733729255023,
		       -0.42867123534264912, 0.13960012162303942,
		       -0.92853609529995151, -2.27570530207327471,
		       0.14015665898430160, -1.23323076638605689,
		       -1.22214475767538078, 0.94582116913123992,
		       0.56415816183362599, -0.43994932699296524,
		       -0.86826316097771228, 1.12160747096193947,
		       0.47775932996734077, -0.48169501830610134,
		       -0.84359927194094375, -0.20188546469610644,
		       -0.04766037970098479, 0.35063557445750559,
		       0.97711951185130286, 0.04246092997736389,
		       1.29244595632070625, 0.21059832507337470,
		       -0.21381806598517750, -1.06403948534350534,
		       1.20245168806249003, 0.53432396516618208,
		       0.70487044424229128, -0.61436020472293806,
		       -0.85750606120704653, 1.12037279227729347,
		       -0.81195974597652887, -1.83846345397798761,
		       -0.09485848511656786, -1.06334390416298152,
		       -0.57764175540636331, -1.02554148316137916,
		       -0.26806654261796614, 0.76792928323320253,
		       -1.05712615586139780, -2.16537567919898244,
		       -0.24050990270062647, 1.27517794982694888,
		       -0.77782533567465173, 0.23651227820308746,
		       0.42854336929812065, 0.33136166011040941,
		       0.84760869375581660, -0.75371013514524243,
		       -1.12107070064550096, -1.84210537826957998,
		       1.18392051308360635, 0.62661920686094441,
		       0.45954379362670339, 1.91585950626078194,
		       -1.75991535996645454, 0.49825457526370193,
		       0.37047870830564478, -0.02656235141619351,
		       1.26023975015208212, 0.01219142322443663,
		       -0.33587572846742192, 0.02351500296513544,
		       -0.32099343361390115, 0.12551705150702128,
		       0.10725277299646366, 0.68088838949363450,
		       -0.44310235240845924, 0.88979140842444937,
		       0.35605037334833645, 0.56338056619996169,
		       0.13032997865692836, 0.79333557877045524,
		       1.46803137123626182, -0.18482938011723204,
		       0.02660504923428107, 0.47978831039611108,
		       -0.33025041294944735, 0.68038307633850159,
		       0.24775249090831936, -0.37188345674250434,
		       -0.29172199069036270, 0.90527826691908919,
		       -1.09947668845334801, -0.33715862293871979,
		       -0.77181409395747136, 0.76847473039369063,
		       -0.18415103970799759, -0.12495454759089894,
		       0.02178667914780518, -0.95089430536131636,
		       2.06564036565978215, 0.42608687812818091,
		       1.01944260386378471, 0.86649835512016327,
		       -0.14018958075608845, -0.17510790205099150,
		       -0.15390418997208721 };

  double C3[n2*m2] = { -1.57867489014846996, 1.16200351462625107,
		       -0.23854682046584150, 1.91410777516692776,
		       2.76806783859857441, 3.57162190525965961,
		       -3.78876613826226682, 5.73963476759751057,
		       3.03081047255006197, 1.81165261573477454,
		       -1.60130923260925995, 7.73438507470749403,
		       1.52279602087881982, 1.57328812957276232,
		       3.98276622035952244, 4.45322366374469070,
		       -1.92682657026162962, -4.96487940401339589,
		       -4.57771218942694080, -8.17482603002922303,
		       4.81217532932217207, -0.19094452826275296,
		       -3.31083952688263938, -2.71262415058127537,
		       -2.98463428691841548, 3.40787575631863637,
		       -0.01334974117374859, -4.55955531556936489,
		       0.52106489966034297, 2.94317867536005950,
		       -6.33805046218492585, -0.95827519771633907,
		       2.23039753744714808, -3.30027292837214192,
		       -4.00308496764564392, 0.17729490924176597,
		       1.66543751499376724, -0.69023588955611248,
		       -3.32810664376311971, 0.34530452700379710,
		       -0.99125148707776667, -0.69931282086587576,
		       -4.37690464934460177, -5.77348528014008711,
		       -1.97935401477902229, -0.06406237677383365,
		       -5.45992391038462888, -4.20807077377006244,
		       2.26552980220935440, -1.27257496021087579,
		       -2.47597035040760582, -1.76747953458818907,
		       -4.99951555962103633, -2.21643257008141337,
		       0.63434628816978411, 0.64667450353985689,
		       2.92190836402693144, -5.21485189097242863,
		       4.54252890190327729, -1.34101750421297683,
		       -2.74584977863530089, -1.38819261606339261,
		       0.86143895794059788, 2.72484661370976999,
		       -4.66417962043932466, 1.83038405759309830,
		       4.09362097415811554, -2.32657353392440136,
		       -5.26541985820923131, 4.63918284584130092,
		       4.86394776043184063, -0.79454605291448410,
		       3.66865536386066138, 0.10422574128505087,
		       0.12790088682418949, -0.52268778693478690,
		       1.15237484252242672, 3.29795609466657869,
		       5.39094589469329399, 4.74925091428268953,
		       1.77137174271968378, -1.69631657421264048,
		       0.06526247899642319, -3.88503014857507578,
		       2.09223645765808142, 0.63334239781176160,
		       -1.98241915668063484, -3.18535507156095532,
		       -2.68702559216853487, 4.67877927441546948,
		       -3.82887259976148009, 2.49591403534012191,
		       2.15731376726530533, 5.12311650997225421,
		       5.49540208205868463, -3.68018458712113228,
		       5.62518723532024367, 5.03594669507498338,
		       -2.76522947535893371, 1.28107435744112008,
		       0.36135265457560484, 4.41276131666740135,
		       1.11953077855202121, -1.84006650807349681,
		       1.95233706194443624, 4.87255790165397329,
		       -5.12118260175422346, -2.17586556094873762,
		       4.52016813922102045, -1.00161050101584959,
		       -4.82686264016570554, 2.58645718066244479,
		       2.30892275237229994, 0.28817816775085581,
		       3.52166518740187451, -4.37283813654189579,
		       4.83362977627106361, -1.90759969837514265,
		       -1.60723396665131690, 0.21593929305519763,
		       -1.40831154735614006, 0.12262195801729986,
		       -1.75539262433655829, 1.33520357499951614,
		       5.55618303900418642, 4.85847038793577291,
		       2.92218534691358212, 3.93496478732698440,
		       -2.74984107761395613, 2.58437485718960813,
		       0.78505970901662181, -0.90184883156189954,
		       3.31798585097429966, -5.19410426798371283,
		       5.58668163060660738, 4.85792008576588241,
		       1.60048224024580499, -2.27752347561349655,
		       -4.51589645382620830, 3.04461709037850126,
		       2.90672079274774120, -5.17391187725850621,
		       -1.72984183541924397, 4.14344376854194785,
		       1.10382521193084093, -1.04985795359579637,
		       9.60901483138428780, -8.17983957178516974,
		       -0.87502348585906997, 0.77703124720234884,
		       3.64855815619328361, -0.55545992999811633,
		       3.37849670424906545, -0.77019590505968660,
		       -1.39059423261508619, 0.21389622983727122,
		       1.96559030867902895, 0.91894069393931943,
		       -3.84915015120662707, -0.14947214123138217,
		       5.76836110040199035, 5.25363980753435289,
		       -4.81554662682277002, 1.50586715627076018,
		       1.98582098118763062, 0.56521835184949754,
		       3.32764177445019182, 6.76107323270590577,
		       -0.69656024408541117, -1.55076810854235658,
		       0.92535811142434854, -8.54950014022918303,
		       2.96351670113916921, 2.09168227391720185,
		       -1.80170642818076754, -5.40490391414054638,
		       2.94534442203043012, -0.50061071262799461,
		       -1.59514372960883155, -6.60116624831505039,
		       0.79007850068030239, 0.69413905931628772,
		       -5.17564824712639648, -1.28248796754816441,
		       1.69949369970153152, -3.50098664416261807,
		       2.89478743128781435, 1.05314343490808326,
		       0.43830746940297838, -2.44932764246232448,
		       -5.49086089359603946, -0.83166550061766253,
		       9.60371650898000517, -3.98069721951972300,
		       0.21794009352348453, -6.46273826528124751,
		       -1.68050792528781412, 2.91000502418165530,
		       -13.75325005826355529, 0.14790505221817207,
		       2.38036190000923309, 0.91094929312121098,
		       -5.37199041728767490, 5.72273950582400470,
		       -5.35706520807667452, 2.41196424120804798,
		       2.03324077616661425, 4.48425944555865641,
		       2.64608583422319210, 6.24860530844928075,
		       -3.02402953180801681, -1.72644274027696376,
		       2.73041513179823614, 3.29595709971412276,
		       5.89021211154137703, -0.97076563276892913,
		       4.04672446224802851, 6.09445888048514028,
		       2.56177895991546478, 1.87094606819759401,
		       3.43882382292970723, -1.84611143299154667,
		       -3.09700849656634780, 4.48844921672020636,
		       1.82817497351973679, -0.85987774559335295,
		       -2.00551513722052377, -1.07604182822002237,
		       4.02723750237333533, 3.16703214948158340,
		       -1.47003401466478523, 0.14275762897324928,
		       0.73224528486796803, -3.35950138564005041,
		       2.07674479586591465, -2.95821227827627231,
		       2.64939319597945300, -3.43670715301030105,
		       -6.04743451071568927, 0.32157364985670056,
		       3.55693086929407043, 1.80661341253903140,
		       1.49161003425708727, -6.26903951860941433,
		       1.88459255603943010, 0.68309973912776178,
		       -1.30918358541226088, 2.82746766943860139,
		       -3.67343280753794099, 3.91613629442515876,
		       -2.46393524485529802, 1.70910290566557177,
		       -0.66338016814740297, -5.14629674818920257,
		       -2.21021018046750362, -3.47116094030537248,
		       0.36406852246054838, -5.49252136377726075,
		       3.64647312165497040, 0.58802602206329602,
		       0.41885360795351001, 2.48738816336166169,
		       4.76003190498936846, 2.23160306565798283,
		       -0.77479162177109473, 2.01537781150356921,
		       -8.87003615502961829, 3.17395003167756906,
		       0.39648912196233288, 0.31649590959790769,
		       0.82082601582671144, -2.55213969481162284,
		       -3.83339067451265114, 0.45867605813626733,
		       -2.08289263745008446, 1.80954515779884040,
		       -5.13607098628819259, 2.57865602228846225,
		       1.89786702904171434, 4.13133152066360498,
		       -0.34903181550686829, 6.24049401495082190,
		       -2.05539981587329335, -1.91030751213821315,
		       -1.25555689895014533, 0.59177976404403188,
		       -1.26996102299965963, -1.44525410879700700,
		       -4.95736174344765246, 3.22930177341965985,
		       1.18597980842937889, -4.90943603511472215,
		       1.55927795957236004, 1.86008024261107408,
		       4.87049234511021201, -5.94607105710878070,
		       0.23439853546196829, 1.04414358703631360,
		       -4.26472696523345984, 8.68141036863939064,
		       1.62416692805219287, -0.33188846067747158,
		       -4.07106789498412347, -1.99723352425991041,
		       -6.20327679950977728, 4.19969111959342367,
		       0.74259205709887521, -5.11678270459288154,
		       -3.45944905641093348, 1.65845971260675329,
		       2.92282420947534627, -1.03717214988839079,
		       -3.16876281303064466, 3.49355836779253970,
		       4.94402598748415301, -6.16574196306213285,
		       0.45261959460468210, 4.30589675340044131,
		       -2.40246296148981164, -0.80388495982019370,
		       2.17852581296403747, -0.01746554572289913,
		       0.12074001064381444 };

  double A4[n2*d2];
  double B4[m2*d2];
  double C4[n2*m2];
  array::mov(n2*d2, A3, A4);
  array::mov(m2*d2, B3, B4);
  array::matmulT(n2, m2, d2, A4, B4, C4);

  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < d2; ++j) {
      int idx = i*d2 + j;
      REQUIRE(A3[idx] == A4[idx]);
    }
  }
  REQUIRE(array::equal(n2*d2,A3,A4));

  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < d2; ++j) {
      int idx = i*d2 + j;
      REQUIRE(B3[idx] == B4[idx]);
    }
  }
  REQUIRE(array::equal(m2*d2,B3,B4));

  for (int i = 0; i < n2; ++i) {
    for (int j = 0; j < m2; ++j) {
      int idx = i*m2 + j;
      REQUIRE(std::abs(C3[idx] - C4[idx]) < 1e-14);
    }
  }
  REQUIRE(!array::equal(n2*m2,C3,C4));
  REQUIRE(array::almostEqual(n2*m2,C3,C4,1e-14));
}



TEST_CASE( "test_matmul2", "[array][matmul]" ) {
  const int n1 = 13;
  const int d1 = 11;
  const int m1 = 17;
  float A[n1*d1];
  float B[m1*d1];
  float C1[n1*m1];
  float C2[n1*m1];

  mt19937 rng(11779933);
  normal_distribution<float> dist;
  for (int i = 0; i < n1*d1; ++i)
    A[i] = dist(rng);
  for (int i = 0; i < m1*d1; ++i)
    B[i] = dist(rng);
  
  cblas_sgemm(CblasRowMajor, // Order: CBLAS_ORDER, 
	      CblasNoTrans,  // TransA: CBLAS_TRANSPOSE, 
              CblasTrans,    // TransB: CBLAS_TRANSPOSE, 
              n1,            // M: Int32, 
              m1,            // N: Int32, 
              d1,            // K: Int32, 
              1.0,           // alpha: Float,
              A,             // A: UnsafePointer<Float>!, 
              d1,            // lda: Int32, 
              B,             // B: UnsafePointer<Float>!, 
              d1,            // ldb: Int32, 
              0.0,           // beta: Float, 
              C1,            // C: UnsafeMutablePointer<Float>!, 
              m1             // ldc: Int32)
	      );
  array::matmulT(n1,m1,d1,A,B,C2);
  for (int i = 0; i < n1*m1; ++i) {
    REQUIRE(std::abs(C1[i] - C2[i]) < 1e-6);
  }
  REQUIRE(array::almostEqual(n1*m1,C1,C2,1e-6f));
}



TEST_CASE( "test_matmul3", "[array][matmul]" ) {
  const int n1 = 23;
  const int d1 = 19;
  const int m1 = 29;
  double A[n1*d1];
  double B[m1*d1];
  double C1[n1*m1];
  double C2[n1*m1];

  mt19937 rng(11779933);
  normal_distribution<double> dist;
  for (int i = 0; i < n1*d1; ++i)
    A[i] = dist(rng);
  for (int i = 0; i < m1*d1; ++i)
    B[i] = dist(rng);
  
  cblas_dgemm(CblasRowMajor, // Order: CBLAS_ORDER, 
	      CblasNoTrans,  // TransA: CBLAS_TRANSPOSE, 
              CblasTrans,    // TransB: CBLAS_TRANSPOSE, 
              n1,            // M: Int32, 
              m1,            // N: Int32, 
              d1,            // K: Int32, 
              1.0,           // alpha: Float,
              A,             // A: UnsafePointer<Float>!, 
              d1,            // lda: Int32, 
              B,             // B: UnsafePointer<Float>!, 
              d1,            // ldb: Int32, 
              0.0,           // beta: Float, 
              C1,            // C: UnsafeMutablePointer<Float>!, 
              m1             // ldc: Int32)
	      );
  array::matmulT(n1,m1,d1,A,B,C2);
  for (int i = 0; i < n1*m1; ++i) {
    REQUIRE(std::abs(C1[i] - C2[i]) < 1e-14);
  }
  REQUIRE(array::almostEqual(n1*m1,C1,C2,1e-14));
}



TEST_CASE( "test_matmul4", "[array][matmul]" ) {
  const int n = 1009;
  const int d = 997;
  const int m = 1129;
  float* A = allocator.allocate<float>(n*d);
  float* B = allocator.allocate<float>(m*d);
  float* C1 = allocator.allocate<float>(n*m);
  float* C2 = allocator.allocate<float>(n*m);

  mt19937 rng(4817);
  normal_distribution<float> dist;
  for (int i = 0; i < n*d; ++i)
    A[i] = dist(rng);
  for (int i = 0; i < m*d; ++i)
    B[i] = dist(rng);

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  array::matmulT(n,m,d,A,B,C1);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "array::matmulT float32 " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  array::gemm(false, true, n, m, d, 1.0f, A, B, 0.0f, C2);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  diff = timerOff();
  cerr << "array::gemm float32 " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  
  for (int i = 0; i < n*m; ++i) {
    REQUIRE(std::abs(C1[i] - C2[i]) < 1e-3);
  }
  REQUIRE(array::almostEqual(n*m,C1,C2,1e-3f));

  allocator.free(A);
  allocator.free(B);
  allocator.free(C1);
  allocator.free(C2);
}



TEST_CASE( "test_matmul5", "[array][matmul]" ) {
  const int n = 827;
  const int d = 953;
  const int m = 1867;
  double* A = allocator.allocate<double>(n*d);
  double* B = allocator.allocate<double>(m*d);
  double* C1 = allocator.allocate<double>(n*m);
  double* C2 = allocator.allocate<double>(n*m);

  mt19937 rng(4817);
  normal_distribution<double> dist;
  for (int i = 0; i < n*d; ++i)
    A[i] = dist(rng);
  for (int i = 0; i < m*d; ++i)
    B[i] = dist(rng);

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  array::matmulT(n,m,d,A,B,C1);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "array::matmulT float64" << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  array::gemm(false, true, n, m, d, 1.0, A, B, 0.0, C2);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  diff = timerOff();
  cerr << "array::gemm float64" <<  parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  
  for (int i = 0; i < n*m; ++i) {
    REQUIRE(std::abs(C1[i] - C2[i]) < 1e-12);
  }
  REQUIRE(array::almostEqual(n*m,C1,C2,1e-12));

  allocator.free(A);
  allocator.free(B);
  allocator.free(C1);
  allocator.free(C2);
}



TEST_CASE( "test_matmul6", "[array][matmul]" ) {
  const int n = 2207;
  const int d = 1151;
  float* A = allocator.allocate<float>(n*d);
  float* x = allocator.allocate<float>(d);
  float* y1 = allocator.allocate<float>(n);
  float* y2 = allocator.allocate<float>(n);

  mt19937 rng(2938);
  normal_distribution<float> dist;
  for (int i = 0; i < n*d; ++i)
    A[i] = dist(rng);
  for (int i = 0; i < d; ++i)
    x[i] = dist(rng);

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  array::matmulT(n,1,d,A,x,y1);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr <<  "array::matmulT for mat-vec float32 " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  array::gemv(n, d, 1.0f, A, x, 0.0f, y2);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  diff = timerOff();
  cerr <<  "array::gemv for mat-vec float32 " << parseDuration(diff) << endl; 
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  
  for (int i = 0; i < n; ++i) {
    REQUIRE(std::abs(y1[i] - y2[i]) < 1e-3);
  }
  REQUIRE(array::almostEqual(n,y1,y2,1e-3f));

  allocator.free(A);
  allocator.free(x);
  allocator.free(y1);
  allocator.free(y2);
}



TEST_CASE( "test_matmul7", "[array][matmul]" ) {
  const int n = 2239;
  const int d = 1193;
  double* A = allocator.allocate<double>(n*d);
  double* x = allocator.allocate<double>(d);
  double* y1 = allocator.allocate<double>(n);
  double* y2 = allocator.allocate<double>(n);

  mt19937 rng(2938);
  normal_distribution<double> dist;
  for (int i = 0; i < n*d; ++i)
    A[i] = dist(rng);
  for (int i = 0; i < d; ++i)
    x[i] = dist(rng);

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  array::matmulT(n,1,d,A,x,y1);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr <<  "array::matmulT for mat-vec float64 " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  array::gemv(n, d, 1.0, A, x, 0.0, y2);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  diff = timerOff();
  cerr <<  "array::gemv for mat-vec float64 "  << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  
  for (int i = 0; i < n; ++i) {
    REQUIRE(std::abs(y1[i] - y2[i]) < 1e-12);
  }
  REQUIRE(array::almostEqual(n,y1,y2,1e-12));

  allocator.free(A);
  allocator.free(x);
  allocator.free(y1);
  allocator.free(y2);
}


TEST_CASE( "test_ger1", "[array]" ) {
  const int n = 1250;
  const int m = 500;
  float* Q = allocator.allocate<float>(m);
  float* X = allocator.allocate<float>(n);  
  float* A1 = allocator.allocate<float>(m*n);
  float* A2 = allocator.allocate<float>(m*n);
  
  mt19937 rng(2938);
  normal_distribution<float> dist;

  for (int i = 0; i < m; ++i)
    Q[i] = dist(rng);
  for (int i = 0; i < n; ++i)
    X[i] = dist(rng);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      A2[i*n + j] = Q[i]*X[j];
  for (int i = 0; i < n*m; ++i)
    A1[i] = 0;
  
  array::ger(m, n, 1.0f, Q, X, A1);

  for (int i = 0; i < n*m; ++i) {
    if (A1[i] != A2[i]) {
      cerr << "A1[" << i << "] = " << A1[i] << " != " << A2[i] << " = A2[" << i
	   << "]" << endl;
    }
    REQUIRE(A1[i] == A2[i]);
  }
  
  REQUIRE(array::equal(n*m,A1,A2));

  allocator.free(Q);
  allocator.free(X);
  allocator.free(A1);
  allocator.free(A2);
}



TEST_CASE( "test_ger2", "[array]" ) {
  const int n = 1250;
  const int m = 500;
  double* Q = allocator.allocate<double>(m);
  double* X = allocator.allocate<double>(n);
  double* A1 = allocator.allocate<double>(m*n);
  double* A2 = allocator.allocate<double>(m*n);
  
  mt19937 rng(2938);
  normal_distribution<double> dist;

  for (int i = 0; i < m; ++i)
    Q[i] = dist(rng);
  for (int i = 0; i < n; ++i)
    X[i] = dist(rng);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      A2[i*n + j] = Q[i]*X[j];
  for (int i = 0; i < n*m; ++i)
    A1[i] = 0;
  array::ger(m, n, 1.0, Q, X, A1);

  REQUIRE(array::equal(n*m,A1,A2));

  allocator.free(Q);
  allocator.free(X);
  allocator.free(A1);
  allocator.free(A2);
}



TEST_CASE( "test_rowwise_mean", "[array]" ) {
  const int n = 1476;
  const int d = 112;
  const int m = 852;
  float A[n*d];
  double B[m*d];
  float C1[n];
  float C2[n];
  double D1[m];
  double D2[m];

  mt19937 rng(2938);
  normal_distribution<double> dist;

  for (int i = 0; i < n*d; ++i)
    A[i] = dist(rng);
  for (int i = 0; i < m*d; ++i)
    B[i] = dist(rng);

  for (int i = 0; i < n; ++i) {
    C1[i] = 0;
    for (int j = 0; j < d; ++j) {
      C1[i] += A[i*d+j];
    }
    C1[i] /= d;
  }

  array::rowwiseMean(n, d, A, C2);

  for (int i = 0; i < n; ++i) {
    REQUIRE(C1[i] == C2[i]);
  }
  
  REQUIRE(array::equal(n,C1,C2));
  
  for (int i = 0; i < m; ++i) {
    D1[i] = 0;
    for (int j = 0; j < d; ++j) {
      D1[i] += B[i*d+j];
    }
    D1[i] /= d;
  }

  array::rowwiseMean(m, d, B, D2);
  
  for (int i = 0; i < m; ++i) {
    REQUIRE(D1[i] == D2[i]);
  }
  
  REQUIRE(array::equal(m,D1,D2));
}



TEST_CASE( "test_compute_dists1", "[array]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);
  REQUIRE(X == data);
  REQUIRE(Y == data + nX*d);

  const int samples = 567;
  vector<float> scratch(samples*(d+2) + 1 /*d + samples*d + samples*/);

  vector<float> ref(nY * samples);
  vector<float> Z1(nY*samples);
  vector<float> Z2(nY*samples);
  vector<float> Z3(nY*samples);
  mt19937 rng;
  uniform_int_distribution<uint32_t> ud(0,nX-1);
  vector<uint32_t> indices(samples*nY);
  for (int i = 0; i < samples*nY; ++i)
    indices[i] = ud(rng);

  for (Metric M : { Metric::EUCLIDEAN, Metric::TAXICAB }) {
    for (int i = 0; i < nY; ++i) {
      float* q = Y + i*d;
      for (int j = 0; j < samples; ++j) {
	float dist = 0;
	uint32_t idx = indices[i*samples+j];
	float* x = X + idx*d;
	for (int k = 0; k < d; ++k) {
	  float diff = q[k] - x[k];
	  switch(M) {
	  case Metric::EUCLIDEAN:
	    dist += diff*diff;
	    break;
	  case Metric::TAXICAB:
	    dist += std::abs(diff);
	    break;
	  }
	}
	if (M == Metric::EUCLIDEAN)
	  dist = std::sqrt(dist);
	ref[i*samples + j] = dist;
      }
    }
    
    for (int i = 0; i < nY; ++i) {
      float* q = Y + i*d;
      uint32_t* idx = &indices[0] + i*samples;
      float* Z = &Z1[0] + i*samples;
      if (M == Metric::EUCLIDEAN)
	array::computeDistsSimple<float,uint32_t,Metric::EUCLIDEAN>(samples, d, X, q, idx, Z, &scratch[0]);
      if (M == Metric::TAXICAB)
	array::computeDistsSimple<float,uint32_t,Metric::TAXICAB>(samples, d, X, q, idx, Z, &scratch[0]);
      
    }


    for (int i = 0; i < nY*samples; ++i) {
      REQUIRE(std::abs(ref[i] - Z1[i]) < 2e-4);
    }
    REQUIRE(array::almostEqual(nY*samples, &ref[0], &Z1[0], 2e-4f));

    if (M == Metric::EUCLIDEAN) {
      for (int i = 0; i < nY; ++i) {
	float* q = Y + i*d;
	uint32_t* idx = &indices[0] + i*samples;
	float* Z = &Z2[0] + i*samples;
	array::computeDistsEuclideanMatmul<float,uint32_t>(samples, d, X, q, idx, Z, &scratch[0]);
      }

      for (int i = 0; i < nY*samples; ++i) {
	REQUIRE(std::abs(ref[i] - Z2[i]) < 1e-3);
      }
      REQUIRE(array::almostEqual(nY*samples, &ref[0], &Z2[0], 1e-3f));
    }

    for (int i = 0; i < nY; ++i) {
      float* q = Y + i*d;
      uint32_t* idx = &indices[0] + i*samples;
      float* Z = &Z3[0] + i*samples;
      if (M == Metric::EUCLIDEAN)
	array::computeDists<float,uint32_t,Metric::EUCLIDEAN>(samples, d, X, q, idx, Z, &scratch[0]);
      else if (M == Metric::TAXICAB)
	array::computeDists<float,uint32_t,Metric::TAXICAB>(samples, d, X, q, idx, Z, &scratch[0]);
    }

    for (int i = 0; i < nY*samples; ++i) {
      REQUIRE(std::abs(ref[i] - Z3[i]) < 3e-4);
    }
    REQUIRE(array::almostEqual(nY*samples, &ref[0], &Z3[0], 3e-4f));
      REQUIRE(array::equal(nY*samples, &Z1[0], &Z3[0]));
  }

  allocator.free(data);
}



TEST_CASE( "test_compute_dists2", "[array]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);
  REQUIRE(X == data);
  REQUIRE(Y == data + nX*d);

  const int samples = 567;
  vector<double> scratch(samples*(d+2) + 1 /*d + samples*d + samples*/);

  vector<double> ref(nY * samples);
  vector<double> Z1(nY*samples);
  vector<double> Z2(nY*samples);
  vector<double> Z3(nY*samples);
  mt19937 rng;
  uniform_int_distribution<uint32_t> ud(0,nX-1);
  vector<uint32_t> indices(samples*nY);
  for (int i = 0; i < samples*nY; ++i)
    indices[i] = ud(rng);

  for (Metric M : { Metric::EUCLIDEAN, Metric::TAXICAB }) {
    for (int i = 0; i < nY; ++i) {
      double* q = Y + i*d;
      for (int j = 0; j < samples; ++j) {
	double dist = 0;
	uint32_t idx = indices[i*samples+j];
	double* x = X + idx*d;
	for (int k = 0; k < d; ++k) {
	  double diff = q[k] - x[k];
	  switch(M) {
	  case Metric::EUCLIDEAN:
	    dist += diff*diff;
	    break;
	  case Metric::TAXICAB:
	    dist += std::abs(diff);
	    break;
	  }
	}
	if (M == Metric::EUCLIDEAN)
	  dist = std::sqrt(dist);
	ref[i*samples + j] = dist;
      }
    }
    
    for (int i = 0; i < nY; ++i) {
      double* q = Y + i*d;
      uint32_t* idx = &indices[0] + i*samples;
      double* Z = &Z1[0] + i*samples;
      if (M == Metric::EUCLIDEAN)
	array::computeDistsSimple<double,uint32_t,Metric::EUCLIDEAN>(samples, d, X, q, idx, Z, &scratch[0]);
      if (M == Metric::TAXICAB)
	array::computeDistsSimple<double,uint32_t,Metric::TAXICAB>(samples, d, X, q, idx, Z, &scratch[0]);
      
    }


    for (int i = 0; i < nY*samples; ++i) {
      REQUIRE(std::abs(ref[i] - Z1[i]) < 1e-12);
    }
    REQUIRE(array::almostEqual(nY*samples, &ref[0], &Z1[0], 1e-12));

    if (M == Metric::EUCLIDEAN) {
      for (int i = 0; i < nY; ++i) {
	double* q = Y + i*d;
	uint32_t* idx = &indices[0] + i*samples;
	double* Z = &Z2[0] + i*samples;
	array::computeDistsEuclideanMatmul<double,uint32_t>(samples, d, X, q, idx, Z, &scratch[0]);
      }

      for (int i = 0; i < nY*samples; ++i) {
	REQUIRE(std::abs(ref[i] - Z2[i]) < 1e-12);
      }
      REQUIRE(array::almostEqual(nY*samples, &ref[0], &Z2[0], 1e-12));
    }

    for (int i = 0; i < nY; ++i) {
      double* q = Y + i*d;
      uint32_t* idx = &indices[0] + i*samples;
      double* Z = &Z3[0] + i*samples;
      if (M == Metric::EUCLIDEAN)
	array::computeDists<double,uint32_t,Metric::EUCLIDEAN>(samples, d, X, q, idx, Z, &scratch[0]);
      else if (M == Metric::TAXICAB)
	array::computeDists<double,uint32_t,Metric::TAXICAB>(samples, d, X, q, idx, Z, &scratch[0]);
    }

    for (int i = 0; i < nY*samples; ++i) {
      REQUIRE(std::abs(ref[i] - Z3[i]) < 1e-12);
    }
    REQUIRE(array::almostEqual(nY*samples, &ref[0], &Z3[0], 1e-12));
      REQUIRE(array::equal(nY*samples, &Z1[0], &Z3[0]));
  }

  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_simple1", "[naivekde]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);
  float scratch[9];
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);
  REQUIRE(X == data);
  REQUIRE(Y == data + nX*d);

  float mu[2174];
  
  const float h = 16;
  
  for (int i = 0; i < nY; ++i) {
    mu[i] = 0;
    float* q = Y + i*d;
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = 0;
      for (int k = 0; k < d; ++k) {
	float tmp = x[k] - q[k];
	dist += tmp*tmp;
      }
      dist = std::sqrt(dist);
      mu[i] += std::exp(-dist/h);
    }
    mu[i] /= nX;
  }

  float Z1[2174];
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    kdeEuclideanSimple(nX, 1, d, h, X, q, Z1 + i, scratch, Kernel::EXPONENTIAL);
    REQUIRE(std::abs(Z1[i]-mu[i]) < 1e-6);
  }
  REQUIRE(!array::equal(nY, Z1, mu));
  REQUIRE(array::almostEqual(nY, Z1, mu, 1e-6f));

  float Z2[2174];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  kdeEuclideanSimple(nX, nY, d, h, X, Y, Z2, scratch, Kernel::EXPONENTIAL);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde simple euclidean exponential float32 "  << parseDuration(diff) << endl; // expected: 63 ms
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  REQUIRE(!array::equal(nY, Z2, mu));
  REQUIRE(array::almostEqual(nY, Z2, mu, 1e-6f));
  REQUIRE(array::equal(nY, Z2, Z1));

  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_simple2", "[naivekde]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  double scratch[9];
  
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);
  REQUIRE(X == data);
  REQUIRE(Y == data + nX*d);

  double mu[2173];
  double Z1[2173];
  const double h = 17;

 for (int i = 0; i < nX; ++i) {
    mu[i] = 0;
    double* q = X + i*d;
    for (int j = 0; j < nY; ++j) {
      double* y = Y + j*d;
      double dist = 0;
      for (int k = 0; k < d; ++k) {
	double tmp = y[k] - q[k];
	dist += tmp*tmp;
      }
      dist = std::sqrt(dist);
      mu[i] += std::exp(-dist/h);
    }
    mu[i] /= nY;
  }
  
  for (int i = 0; i < nX; ++i) {
    double* q = X + i*d;
    kdeEuclideanSimple(nY, 1, d, h, Y, q, Z1 + i, scratch, Kernel::EXPONENTIAL);
    REQUIRE(std::abs(Z1[i]-mu[i]) < 1e-16);
  }
  REQUIRE(!array::equal(nX, Z1, mu));
  REQUIRE(array::almostEqual(nX, Z1, mu, 1e-16));

  double Z2[2173];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  kdeEuclideanSimple(nY, nX, d, h, Y, X, Z2, scratch, Kernel::EXPONENTIAL);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde simple euclidean exponential float64 "  << parseDuration(diff) << endl; // expected: 78 ms
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  REQUIRE(!array::equal(nX, Z2, mu));
  REQUIRE(array::almostEqual(nX, Z2, mu, 1e-16));
  REQUIRE(array::equal(nX, Z1, Z2));

  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_simple3", "[naivekde]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);
  REQUIRE(X == data);
  REQUIRE(Y == data + nX*d);

  float mu[2174];
  const float h = 21;
  float scratch[9];
  
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    mu[i] = 0;
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = array::euclideanDistance(d, q, x, scratch);
      mu[i] += std::exp(-dist*dist/2/h/h);
    }
    mu[i] /= nX;
  }
  

  float Z1[2174];
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    kdeEuclideanSimple(nX, 1, d, h, X, q, Z1 + i, scratch, Kernel::GAUSSIAN);
    REQUIRE(std::abs(Z1[i]-mu[i]) < 1e-6);
  }
  REQUIRE(array::equal(nY, Z1, mu));

  float Z2[2174];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  kdeEuclideanSimple(nX, nY, d, h, X, Y, Z2, scratch, Kernel::GAUSSIAN);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde simple euclidean gaussian " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  REQUIRE(array::equal(nY, Z2, mu));
  REQUIRE(array::equal(nY, Z2, Z1));

  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_simple4", "[naivekde]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  double scratch[9];
  
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);
  REQUIRE(X == data);
  REQUIRE(Y == data + nX*d);

  const double h = 32;
  double mu[2173];
  for (int i = 0; i < nX; ++i) {
    double* q = X + i*d;
    mu[i] = 0;
    for (int j = 0; j < nY; ++j) {
      double* y = Y + j*d;
      double dist = array::euclideanDistance(d, q, y, scratch);
      mu[i] += std::exp(-dist*dist/2/h/h);
    }
    mu[i] /= nY;
  }

  
  
  double Z1[2173];
  for (int i = 0; i < nX; ++i) {
    double* q = X + i*d;
    kdeEuclideanSimple(nY, 1, d, h, Y, q, Z1 + i, scratch, Kernel::GAUSSIAN);
    REQUIRE(std::abs(Z1[i]-mu[i]) < 1e-16);
  }
  REQUIRE(array::equal(nX, Z1, mu));

  double Z2[2173];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  kdeEuclideanSimple(nY, nX, d, h, Y, X, Z2, scratch, Kernel::GAUSSIAN);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde simple euclidean gaussian " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  REQUIRE(array::equal(nX, Z2, mu));
  REQUIRE(array::equal(nX, Z1, Z2));

  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_matmul1", "[naivekde]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  float XSqNorms[2173];
  array::sqNorm(nX, d, X, XSqNorms);
  float* scratch = allocator.allocate<float>(2173*2174+2174+2174);
  REQUIRE(nX*nY + nY + std::max(nX,nY) == 2173*2174+2174+2174);
  
  float mu[2174];
  float h = 16;
  kdeEuclideanSimple(nX, nY, d, h, X, Y, mu, scratch, Kernel::EXPONENTIAL);
  
  float Z1[2174];
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    kdeEuclideanMatmul(nX, 1, d, h, X, q, Z1 + i, XSqNorms, scratch,
		       Kernel::EXPONENTIAL);
    REQUIRE(std::abs(mu[i] - Z1[i]) < 1e-6);
  }
  REQUIRE(!array::equal(nY, mu, Z1));
  REQUIRE(array::almostEqual(nY, mu, Z1, 1e-6f));

  float Z2[2174];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  for (int i = 0; i < nX; ++i)
    XSqNorms[i] = array::sqNorm(d, X + i*d);
  kdeEuclideanMatmul(nX, nY, d, h, X, Y, Z2, XSqNorms, scratch, Kernel::EXPONENTIAL);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde euclidean matmul exponential float32 " << parseDuration(diff) << endl; // expected: 47 ms
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  REQUIRE(!array::equal(nY, mu, Z2));
  REQUIRE(array::almostEqual(nY, mu, Z2, 1e-6f));

  for (int i = 0; i < nY; ++i) {
    REQUIRE(std::abs(Z1[i] - Z2[i]) < 1e-7);
  }
    
  
  REQUIRE(array::almostEqual(nY, Z1, Z2, 1e-7f));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_matmul2", "[naivekde]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  double YSqNorms[2174];
  for (int i = 0; i < nY; ++i)
    YSqNorms[i] = array::sqNorm(d, Y + i*d);
  double* scratch = allocator.allocate<double>(2173*2174 + 2173 + 2174);
  
  double mu[2173];
  double h = 23;
  kdeEuclideanSimple(nY, nX, d, h, Y, X, mu, scratch, Kernel::EXPONENTIAL);
  
  double Z1[2173];
  for (int i = 0; i < nX; ++i) {
    double* q = X + i*d;
    kdeEuclideanMatmul(nY, 1, d, h, Y, q, Z1 + i, YSqNorms, scratch,
		       Kernel::EXPONENTIAL);
    REQUIRE(std::abs(mu[i] - Z1[i]) < 1e-15);
  }
  REQUIRE(!array::equal(nX, mu, Z1));
  REQUIRE(array::almostEqual(nX, mu, Z1, 1e-15));

  double Z2[2173];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  for (int i = 0; i < nY; ++i)
    YSqNorms[i] = array::sqNorm(d, Y + i*d);
  kdeEuclideanMatmul(nY, nX, d, h, Y, X, Z2, YSqNorms, scratch, Kernel::EXPONENTIAL);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde euclidean matmul exponential float64 " << parseDuration(diff) << endl; // expected: 57 ms
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  REQUIRE(!array::equal(nX, mu, Z2));

  for (int i = 0; i < nX; ++i) {
    REQUIRE(std::abs(Z1[i] - Z2[i]) < 1e-16);
  }
  REQUIRE(array::almostEqual(nX, Z1, Z2, 1e-16));
  
  REQUIRE(array::almostEqual(nX, mu, Z2, 1e-15));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_matmul3", "[naivekde]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  float XSqNorms[2173];
  array::sqNorm(nX, d, X, XSqNorms);
  float* scratch = allocator.allocate<float>(2173*2174+2174+2174);
  REQUIRE(nX*nY + nY + std::max(nX,nY) == 2173*2174+2174+2174);
  
  float mu[2174];
  float h = 16;
  kdeEuclideanSimple(nX, nY, d, h, X, Y, mu, scratch, Kernel::GAUSSIAN);
  
  float Z1[2174];
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    kdeEuclideanMatmul(nX, 1, d, h, X, q, Z1 + i, XSqNorms, scratch,
		       Kernel::GAUSSIAN);
    REQUIRE(std::abs(mu[i] - Z1[i]) < 1e-6);
  }
  REQUIRE(!array::equal(nY, mu, Z1));
  REQUIRE(array::almostEqual(nY, mu, Z1, 1e-6f));

  float Z2[2174];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  for (int i = 0; i < nX; ++i)
    XSqNorms[i] = array::sqNorm(d, X + i*d);
  kdeEuclideanMatmul(nX, nY, d, h, X, Y, Z2, XSqNorms, scratch, Kernel::GAUSSIAN);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde euclidean matmul gaussian float32" << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  REQUIRE(!array::equal(nY, mu, Z2));
  REQUIRE(array::almostEqual(nY, mu, Z2, 1e-6f));

  for (int i = 0; i < nY; ++i) {
    REQUIRE(std::abs(Z1[i] - Z2[i]) < 1e-7);
  }
    
  
  REQUIRE(array::almostEqual(nY, Z1, Z2, 1e-7f));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_matmul4", "[naivekde]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  double YSqNorms[2174];
  for (int i = 0; i < nY; ++i)
    YSqNorms[i] = array::sqNorm(d, Y + i*d);
  double* scratch = allocator.allocate<double>(2173*2174 + 2173 + 2174);
  
  double mu[2173];
  double h = 33;
  kdeEuclideanSimple(nY, nX, d, h, Y, X, mu, scratch, Kernel::GAUSSIAN);
  
  double Z1[2173];
  for (int i = 0; i < nX; ++i) {
    double* q = X + i*d;
    kdeEuclideanMatmul(nY, 1, d, h, Y, q, Z1 + i, YSqNorms, scratch,
		       Kernel::GAUSSIAN);
    REQUIRE(std::abs(mu[i] - Z1[i]) < 1e-15);
  }
  REQUIRE(!array::equal(nX, mu, Z1));
  REQUIRE(array::almostEqual(nX, mu, Z1, 1e-15));

  double Z2[2173];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  for (int i = 0; i < nY; ++i)
    YSqNorms[i] = array::sqNorm(d, Y + i*d);
  kdeEuclideanMatmul(nY, nX, d, h, Y, X, Z2, YSqNorms, scratch, Kernel::GAUSSIAN);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde euclidean matmul gaussian float64 " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  REQUIRE(!array::equal(nX, mu, Z2));

  for (int i = 0; i < nX; ++i) {
    REQUIRE(std::abs(Z1[i] - Z2[i]) < 3e-16);
  }
  REQUIRE(array::almostEqual(nX, Z1, Z2, 3e-16));
  
  REQUIRE(array::almostEqual(nX, mu, Z2, 1e-15));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_matmul5", "[naivekde]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  vector<float> XSqNorms(nX);
  array::sqNorm(nX, d, X, &XSqNorms[0]);
  vector<float> scratch(nX*nY + nY + std::max(nX,nY));
  vector<float> mu(nY);
  vector<float> Z(nY);
  float h = 27;

  const float DELTA = 1e-6f;
  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN }) {
    kdeEuclideanMatmul(nX, nY, d, h, X, Y, &mu[0], &XSqNorms[0], &scratch[0],
		       kernel);
    for (int i = 0; i < nY; ++i) {
      float* q = Y + i*d;
      Z[i] = kdeEuclideanMatmul(nX, d, h, X, q, &XSqNorms[0], &scratch[0], kernel);
    }

    for (int i = 0; i < nY; ++i) {
      REQUIRE(std::abs(mu[i] - Z[i]) < DELTA);
    }
    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));
  }
	
  allocator.free(data);
}



TEST_CASE( "test_kde_euclidean_matmul6", "[naivekde]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  vector<double> XSqNorms(nX);
  array::sqNorm(nX, d, X, &XSqNorms[0]);
  vector<double> scratch(nX*nY + nY + std::max(nX,nY));
  vector<double> mu(nY);
  vector<double> Z(nY);
  double h = 32;

  const double DELTA = 1e-15;
  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN }) {
    kdeEuclideanMatmul(nX, nY, d, h, X, Y, &mu[0], &XSqNorms[0], &scratch[0],
		       kernel);
    for (int i = 0; i < nY; ++i) {
      double* q = Y + i*d;
      Z[i] = kdeEuclideanMatmul(nX, d, h, X, q, &XSqNorms[0], &scratch[0], kernel);
    }

    for (int i = 0; i < nY; ++i) {
      REQUIRE(std::abs(mu[i] - Z[i]) < DELTA);
    }
    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));
  }
	
  allocator.free(data);
}



TEST_CASE( "test_kde_taxicab1", "[naivekde]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  float* scratch = allocator.allocate<float>(2173*2174+9);
  REQUIRE(nX*nY + nX == 2173*2174+2173);
  
  float mu[2174];
  float h = 17;
  for (int i = 0; i < nY; ++i) {
    mu[i] = 0;
    float* q = Y + i*d;
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = 0;
      for (int k = 0; k < d; ++k)
	dist += std::abs(x[k]-q[k]);
      mu[i] += std::exp(-dist/h);
    }
    mu[i] /= nX;
  }
   
  float Z1[2174];
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    kdeTaxicab(nX, 1, d, h, X, q, Z1 + i, scratch, Kernel::LAPLACIAN);
  }

  for (int i = 0; i < nY; ++i) {
    float relError = relativeError(Z1[i], mu[i]);
    float absError = std::abs(Z1[i] - mu[i]);
    REQUIRE(relError < 2e-6);
    REQUIRE(absError < 2e-9);
  }
  REQUIRE(!array::equal(nY, mu, Z1));
  REQUIRE(array::almostEqual(nY, mu, Z1, 2e-9f));

  float Z2[2174];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  kdeTaxicab(nX, nY, d, h, X, Y, Z2, scratch, Kernel::LAPLACIAN);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde taxicab laplacian float " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  REQUIRE(!array::equal(nY, mu, Z2));
  REQUIRE(array::almostEqual(nY, mu, Z2, 2e-9f));

  for (int i = 0; i < nY; ++i) {
    REQUIRE(Z1[i] == Z2[i]);
  }
    
  
  REQUIRE(array::equal(nY, Z1, Z2));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_kde_taxicab2", "[naivekde]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);

  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  double* scratch = allocator.allocate<double>(2173*2174+9);
  REQUIRE(nX*nY + d == 2173*2174+9);
  
  double mu[2174];
  double h = 18;
  for (int i = 0; i < nX; ++i) {
    mu[i] = 0;
    double* q = X + i*d;
    for (int j = 0; j < nY; ++j) {
      double* y = Y + j*d;
      double dist = 0;
      for (int k = 0; k < d; ++k)
	dist += std::abs(y[k]-q[k]);
      mu[i] += std::exp(-dist/h);
    }
    mu[i] /= nY;
  }
   
  double Z1[2173];
  for (int i = 0; i < nX; ++i) {
    double* q = X + i*d;
    kdeTaxicab(nY, 1, d, h, Y, q, Z1 + i, scratch, Kernel::LAPLACIAN);
  }

  for (int i = 0; i < nX; ++i) {
    double relError = relativeError(Z1[i], mu[i]);
    double absError = std::abs(Z1[i] - mu[i]);
    REQUIRE((relError < 3e-15 || (relError == 1.0 && Z1[i] == 0)));
    REQUIRE(absError < 1e-17);
  }
  REQUIRE(!array::equal(nX, mu, Z1));
  REQUIRE(array::almostEqual(nX, mu, Z1, 1e-17));

  double Z2[2173];
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  kdeTaxicab(nY, nX, d, h, Y, X, Z2, scratch, Kernel::LAPLACIAN);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "kde taxicab laplacian double " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  REQUIRE(!array::equal(nX, mu, Z2));
  REQUIRE(array::almostEqual(nX, mu, Z2, 1e-17));

  for (int i = 0; i < nX; ++i) {
    REQUIRE(Z1[i] == Z2[i]);
  }
    
  
  REQUIRE(array::equal(nX, Z1, Z2));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_naive_kde1", "[naivekde]" ) {
  const float h = 24;
  NaiveKde<float> nkde(h, Kernel::EXPONENTIAL);

  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  nkde.fit(nY, d, Y);

  float mu1[2173];
  int samples[2173];

  REQUIRE_THROWS_AS(nkde.query(d-1, Y, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d+1, Y, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, Y, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, Y-d+1, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, Y + 10, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, Y + 100, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, Y + nY*d - 1, mu1, samples), std::invalid_argument);

  float* scratch = allocator.allocate<float>(2173*2174 + 2173 + 2174);
  
  kdeEuclideanSimple(nY, nX, d, h, Y, X, mu1, scratch, Kernel::EXPONENTIAL);
  float YSqNorms[2174];
  array::sqNorm(nY, d, Y, YSqNorms);

  float mu2[2173];
  kdeEuclideanMatmul(nY, nX, d, h, Y, X, mu2, YSqNorms, scratch,
		     Kernel::EXPONENTIAL);

  float Z1[2173];
  for (int i = 0; i < nX; ++i) {
    float* q = X + i*d;
    nkde.query(d, q, Z1 + i, samples + i);
    REQUIRE(std::abs(Z1[i] - mu1[i]) < 1e-6);
    REQUIRE(std::abs(Z1[i] - mu2[i]) < 1e-7);
  }
  for (int s : samples)
    REQUIRE(s == nY);

  float Z2[2173];
  REQUIRE_THROWS_AS(nkde.query(1, d, Y, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(1, d, Y+(nY-1)*d, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(1, d, Y-d+1, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(1, d, Y+nY*d-1, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(nX+1, d, X, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(nX+1, d, X-d+1, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(1, d, X+(nX-1)*d+1, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(nX, d, X+1, Z2, samples), std::invalid_argument);

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  nkde.fit(nY, d, Y);
  nkde.query(nX, d, X, Z2, samples);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "naive kde exponential float32 " << parseDuration(diff) << endl; // expected: 58 ms
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  for (int s : samples)
    REQUIRE(s == nY);

  for (int i = 0; i < nX; ++i) {
    REQUIRE(std::abs(Z1[i]-Z2[i]) < 1e-7);
  }

  REQUIRE(array::almostEqual(nX,Z1,Z2,1e-7f));
  
  REQUIRE(array::equal(nX,Z2,mu2));
  REQUIRE(array::almostEqual(nX,Z2,mu1,1e-6f));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_naive_kde2", "[naivekde]" ) {
  const double h = 32;
  NaiveKde<double> nkde(h, Kernel::EXPONENTIAL);

  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  nkde.fit(nX, d, X);
  
  int samples[2174];
  double mu1[2174];

  REQUIRE_THROWS_AS(nkde.query(d-1, Y, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d+1, Y, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, X, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, X-d+1, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, X + 10, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, X + 100, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, X + nX*d - 1, mu1, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(d, X + (nX-1)*d + 1, mu1, samples), std::invalid_argument);

  double* scratch = allocator.allocate<double>(2174*2173+2174+2174);
  
  kdeEuclideanSimple(nX, nY, d, h, X, Y, mu1, scratch, Kernel::EXPONENTIAL);
  double XSqNorms[2173];
  array::sqNorm(nX, d, X, XSqNorms);

  double mu2[2174];
  kdeEuclideanMatmul(nX, nY, d, h, X, Y, mu2, XSqNorms, scratch,
		     Kernel::EXPONENTIAL);

  double Z1[2174];
  for (int i = 0; i < nY; ++i) {
    double* q = Y + i*d;
    nkde.query(d, q, Z1 + i, samples + i);
    REQUIRE(std::abs(Z1[i] - mu1[i]) < 1e-15);
    REQUIRE(std::abs(Z1[i] - mu2[i]) < 1e-15);
  }
  for (int s : samples)
    REQUIRE(s == nX);

  
  double Z2[2174];
  REQUIRE_THROWS_AS(nkde.query(1, d, X, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(1, d, X+(nX-1)*d, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(1, d, X-d+1, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(1, d, X+nX*d-1, Z2, samples), std::invalid_argument);
  REQUIRE_THROWS_AS(nkde.query(nY, d, Y-1, Z2, samples), std::invalid_argument);

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  nkde.fit(nX, d, X);
  nkde.query(nY, d, Y, Z2, samples);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "naive kde exponential float64 " << parseDuration(diff) << endl; // expected: 70 ms
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  for (int s : samples)
    REQUIRE(s == nX);
  
  for (int i = 0; i < nY; ++i) {
    REQUIRE(std::abs(Z1[i]-Z2[i]) < 1e-15);
  }
  REQUIRE(array::almostEqual(nY,Z1,Z2,1e-15));
  REQUIRE(array::equal(nY,Z2,mu2));
  REQUIRE(array::almostEqual(nY,Z2,mu1,1e-15));
  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_naive_kde3", "[naivekde]" ) {
  const float h = 25;
  NaiveKde<float> nkde(h, Kernel::GAUSSIAN);

  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  nkde.fit(nY, d, Y);

  float mu1[2173];
  float* scratch = allocator.allocate<float>(2173*2174 + 2173 + 2174);
  
  kdeEuclideanSimple(nY, nX, d, h, Y, X, mu1, scratch, Kernel::GAUSSIAN);
  float YSqNorms[2174];
  array::sqNorm(nY, d, Y, YSqNorms);

  float mu2[2173];
  kdeEuclideanMatmul(nY, nX, d, h, Y, X, mu2, YSqNorms, scratch,
		     Kernel::GAUSSIAN);
  int samples[2173];
  float Z1[2173];
  for (int i = 0; i < nX; ++i) {
    float* q = X + i*d;
    nkde.query(d, q, Z1 + i, samples + i);
    REQUIRE(std::abs(Z1[i] - mu1[i]) < 1e-6);
    REQUIRE(std::abs(Z1[i] - mu2[i]) < 2e-7);
  }
  for (int s : samples)
    REQUIRE(s == nY);


  float Z2[2173];

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  nkde.fit(nY, d, Y);
  nkde.query(nX, d, X, Z2, samples);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "naive kde euclidean gaussian " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  for (int s : samples)
    REQUIRE(s == nY);

  for (int i = 0; i < nX; ++i) {
    REQUIRE(std::abs(Z1[i]-Z2[i]) < 2e-7);
  }

  REQUIRE(array::almostEqual(nX,Z1,Z2,2e-7f));
  
  REQUIRE(array::equal(nX,Z2,mu2));
  REQUIRE(array::almostEqual(nX,Z2,mu1,1e-6f));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_naive_kde4", "[naivekde]" ) {
  const double h = 34;
  NaiveKde<double> nkde(h, Kernel::GAUSSIAN);

  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  nkde.fit(nX, d, X);
  
  double mu1[2174];
  double* scratch = allocator.allocate<double>(2174*2173+2174+2174);
  
  kdeEuclideanSimple(nX, nY, d, h, X, Y, mu1, scratch, Kernel::GAUSSIAN);
  double XSqNorms[2173];
  array::sqNorm(nX, d, X, XSqNorms);

  double mu2[2174];
  kdeEuclideanMatmul(nX, nY, d, h, X, Y, mu2, XSqNorms, scratch,
		     Kernel::GAUSSIAN);

  int samples[2174];

  double Z1[2174];
  for (int i = 0; i < nY; ++i) {
    double* q = Y + i*d;
    nkde.query(d, q, Z1 + i, samples + i);
    REQUIRE(std::abs(Z1[i] - mu1[i]) < 1e-15);
    REQUIRE(std::abs(Z1[i] - mu2[i]) < 1e-15);
  }
  for (int s : samples)
    REQUIRE(s == nX);
  
  double Z2[2174];

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  nkde.fit(nX, d, X);
  nkde.query(nY, d, Y, Z2, &samples[0]);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "naive kde euclidean gaussian " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  for (int s : samples)
    REQUIRE(s == nX);

  for (int i = 0; i < nY; ++i) {
    REQUIRE(std::abs(Z1[i]-Z2[i]) < 1e-15);
  }
  REQUIRE(array::almostEqual(nY,Z1,Z2,1e-15));
  REQUIRE(array::equal(nY,Z2,mu2));
  REQUIRE(array::almostEqual(nY,Z2,mu1,1e-15));
  allocator.free(scratch);
  allocator.free(data); 
}



TEST_CASE( "test_naive_kde5", "[naivekde]" ) {
  const float h = 23;
  NaiveKde<float> nkde(h, Kernel::LAPLACIAN);

  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  nkde.fit(nX, d, X);

  float mu1[2174];
  for (int i = 0; i < nY; ++i) {
    mu1[i] = 0;
    float* q = Y + i*d;
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = 0;
      for (int k = 0; k < d; ++k)
	dist += std::abs(x[k]-q[k]);
      mu1[i] += std::exp(-dist/h);
    }
    mu1[i] /= nX;
  }

  float mu2[2174];
  float* scratch = allocator.allocate<float>(2173*2174 + 9); 
  kdeTaxicab(nX, nY, d, h, X, Y, mu2, scratch, Kernel::LAPLACIAN);

  float Z1[2174];
  int samples[2174];
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    nkde.query(d, q, Z1 + i, samples + i);
  }
  for (int s : samples)
    REQUIRE(s == nX);
  for (int i = 0; i < nY; ++i) {
    float relError = relativeError(mu1[i], Z1[i]);
    float absError = std::abs(Z1[i] - mu1[i]);
    // cerr << Z1[i] << " " << mu1[i] << " " <<  relError << " " << absError << endl;
    REQUIRE(relError < 2e-6);
    REQUIRE(absError < 6e-9);
    REQUIRE(Z1[i] == mu2[i]);
  }

  float Z2[2174];

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  nkde.fit(nX, d, X);
  nkde.query(nY, d, Y, Z2, samples);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "naive kde laplacian float " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

  for (int s : samples)
    REQUIRE(s == nX);

  for (int i = 0; i < nY; ++i) {
    REQUIRE(Z1[i] == Z2[i]);
  }

  REQUIRE(array::equal(nY,Z1,Z2));
  REQUIRE(array::equal(nY,Z2,mu2));
  REQUIRE(array::almostEqual(nY,Z2,mu1,6e-9f));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_naive_kde6", "[naivekde]" ) {
  const double h = 24;
  NaiveKde<double> nkde(h, Kernel::LAPLACIAN);

  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  REQUIRE(n == 4347);
  REQUIRE(d == 9);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);
  REQUIRE(nX == 2173);
  REQUIRE(nY == 2174);

  nkde.fit(nX, d, X);

  double mu1[2174];
  for (int i = 0; i < nY; ++i) {
    mu1[i] = 0;
    double* q = Y + i*d;
    for (int j = 0; j < nX; ++j) {
      double* x = X + j*d;
      double dist = 0;
      for (int k = 0; k < d; ++k)
	dist += std::abs(x[k]-q[k]);
      mu1[i] += std::exp(-dist/h);
    }
    mu1[i] /= nX;
  }

  double mu2[2174];
  double* scratch = allocator.allocate<double>(2173*2174 + 9); 
  kdeTaxicab(nX, nY, d, h, X, Y, mu2, scratch, Kernel::LAPLACIAN);

  double Z1[2174];
  int samples[2174];
  for (int i = 0; i < nY; ++i) {
    double* q = Y + i*d;
    nkde.query(d, q, Z1 + i, samples + i);
  }
  for (int s : samples)
    REQUIRE(s == nX);
  for (int i = 0; i < nY; ++i) {
    double relError = relativeError(mu1[i], Z1[i]);
    double absError = std::abs(Z1[i] - mu1[i]);
    REQUIRE(relError < 1e-14);
    REQUIRE(absError < 2e-17);
    REQUIRE(Z1[i] == mu2[i]);
  }

  double Z2[2174];

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  nkde.fit(nX, d, X);
  nkde.query(nY, d, Y, Z2, &samples[0]);
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
  auto diff = timerOff();
  cerr << "naive kde laplacian double " << parseDuration(diff) << endl;
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  for (int s : samples)
    REQUIRE(s == nX);

  for (int i = 0; i < nY; ++i) {
    REQUIRE(Z1[i] == Z2[i]);
  }

  REQUIRE(array::equal(nY,Z1,Z2));
  REQUIRE(array::equal(nY,Z2,mu2));
  REQUIRE(array::almostEqual(nY,Z2,mu1,2e-17));

  allocator.free(scratch);
  allocator.free(data);
}



TEST_CASE( "test_naive_kde7", "[naivekde]" ) {
  const float h = 2;
  NaiveKde<float> nkde(h, Kernel::LAPLACIAN);

  const int n = 200;
  const int m = 500;
  const int d = 784;
  float X[n*d];
  float Q[m*d];
  int samples[m];

  mt19937 rng(1928);
  normal_distribution<float> dist;

  for (int i = 0; i < n*d; ++i)
    X[i] = dist(rng);
  for (int i = 0; i < m*d; ++i)
    Q[i] = dist(rng);
  
  nkde.fit(n, d, X);

  float* scratch = allocator.allocate<float>(n*m+d);
  float mu[m];

  kdeTaxicab(n, m, d, h, X, Q, mu, scratch, Kernel::LAPLACIAN);

  float Z[m];
  nkde.query(m, d, Q, Z, samples);
  for (int s : samples)
    REQUIRE(s == n);
  REQUIRE(array::equal(m, Z, mu));
  allocator.free(scratch);
}



TEST_CASE( "test_naive_kde8", "[naivekde]" ) {
  const double h = 16;
  float* data1;
  double* data2;
  int n, d;
  constructMediumTestSet(&data1, &n, &d);

  float *X1, *Y1;
  double *X2, *Y2;
  int nX, nY;
  constructMediumTestSet(&data2, &n, &d);
  array::splitInHalf(n,d,&nX,&nY,data1,&X1,&Y1);
  array::splitInHalf(n,d,&nX,&nY,data2,&X2,&Y2);

  NaiveKde<float> nkde1(h, Kernel::EXPONENTIAL);
  NaiveKde<double> nkde2(h, Kernel::EXPONENTIAL);

  REQUIRE_NOTHROW(nkde1.resetParameters());
  REQUIRE_THROWS_AS(nkde1.resetParameters(1), invalid_argument);
  REQUIRE_NOTHROW(nkde1.resetParameters(nullopt));
  REQUIRE_THROWS_AS(nkde1.resetParameters(nullopt,1), invalid_argument);
  REQUIRE_THROWS_AS(nkde1.resetParameters(1,2), invalid_argument);
  REQUIRE_NOTHROW(nkde1.resetParameters(nullopt,nullopt));
  
  REQUIRE_NOTHROW(nkde2.resetParameters());
  REQUIRE_THROWS_AS(nkde2.resetParameters(1), invalid_argument);
  REQUIRE_NOTHROW(nkde2.resetParameters(nullopt));
  REQUIRE_THROWS_AS(nkde2.resetParameters(nullopt,1), invalid_argument);
  REQUIRE_THROWS_AS(nkde2.resetParameters(1,2), invalid_argument);
  REQUIRE_NOTHROW(nkde2.resetParameters(nullopt,nullopt));
  
  allocator.free(data1);
  allocator.free(data2);
}



TEST_CASE( "test_linear_scan1", "[linearscan]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  
  LinearScan<float> ls(Metric::EUCLIDEAN);
  REQUIRE_THROWS_AS(ls.fit(0,d,data), std::invalid_argument);
  REQUIRE_THROWS_AS(ls.fit(n,0,data), std::invalid_argument);
  ls.fit(n,d,data);
  vector<int> nns(1);
  vector<float> dists(1);
  vector<int> samples(1);
  ls.query(1, d, 1, data, &nns[0], &dists[0], &samples[0]);
  REQUIRE(nns[0] == 0);
  REQUIRE(dists[0] == 0);
  REQUIRE(samples[0] == n);

  vector<float> XSqNorms(n);
  array::sqNorm(n, d, data, &XSqNorms[0]);

  vector<float> allDists(n*n);
  vector<float> scratch(std::max(d, 2*n));
  array::euclideanSqDistances(n, n, d, data, data, &XSqNorms[0], &allDists[0], &scratch[0]);

  int m = 100;
  int k = 50;
  nns = vector<int>(m*k);
  dists = vector<float>(m*k);
  samples = vector<int>(m);
  ls.query(m, d, k, data, &nns[0], &dists[0], &samples[0]);
  for (int i = 0; i < m; ++i) {
    REQUIRE(nns[i*k] == i);
    REQUIRE(dists[i*k] >= 0);
    REQUIRE((dists[i*k] == 0 || dists[i*k] == std::sqrt(allDists[i*n+i])));
    REQUIRE(samples[i] == n);
  }

  m = 5;
  k = n + 2;
  nns = vector<int>(m*k);
  dists = vector<float>(m*k);
  ls.query(m, d, k, data, &nns[0], &dists[0], &samples[0]);
  for (int i = 0; i < m; ++i) {
    REQUIRE(nns[i*k] == i);
    REQUIRE(nns[i*k + n] == -1);
    REQUIRE(nns[i*k + n + 1] == -1);
    REQUIRE((dists[i*k] == 0 || dists[i*k] == std::sqrt(allDists[i*n + i])));
    REQUIRE(dists[i*k + n] == -1);
    REQUIRE(dists[i*k + n + 1] == -1);
    REQUIRE(samples[i] == n);
  }

  allocator.free(data);
}



TEST_CASE( "test_linear_scan2", "[linearscan]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  
  LinearScan<float> ls(Metric::EUCLIDEAN);
  ls.fit(nX, d, X);

  const float EPSILON = 1e-3f;
  
  int k = 1;
  vector<int> nns(nY*k);
  vector<float> dists(nY*k);
  vector<float> scratch(d);
  vector<int> samples(nY);

  ls.query(nY, d, k, Y, &nns[0], &dists[0], &samples[0]);
  for (int s : samples)
    REQUIRE(s == nX);

  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    int nnIdx;
    float nnDist = 99999999999999;
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = array::euclideanDistance(d, q, x, &scratch[0]);
      if (dist < nnDist) {
	nnDist = dist;
	nnIdx = j;
      }
    }

    int nn;
    float dist;
    int samp;
    ls.query(1, d, 1, q, &nn, &dist, &samp);
    REQUIRE(nns[i] == nn);
    REQUIRE(nn == nnIdx);
    REQUIRE(dists[i] == Approx(dist).epsilon(EPSILON));
    REQUIRE(dist == Approx(nnDist).epsilon(EPSILON));
    REQUIRE(samples[i] == samp);
    REQUIRE(samp == nX);
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = array::euclideanDistance(d, q, x, &scratch[0]);
      if (j != nn)
	REQUIRE(dist > nnDist);
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan3", "[linearscan]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  mt19937 rng(1234);
  normal_distribution<float> nd(0,200);
  for (int i = 0; i < n*d; ++i)
    data[i] += nd(rng);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const float EPSILON = 1e-4f;
  
  LinearScan<float> ls(Metric::EUCLIDEAN);
  ls.fit(nX, d, X);

  int k = 50;
  vector<int> nns(nY*k);
  vector<float> dists(nY*k);
  vector<float> scratch(d);
  vector<int> samples(nY);

  ls.query(nY, d, k, Y, &nns[0], &dists[0], &samples[0]);
  for (int s : samples)
    REQUIRE(s == nX);


  vector<int> nnSingle(k);
  vector<float> distsSingle(k);
  int samplesSingle;
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    ls.query(1, d, k, q, &nnSingle[0], &distsSingle[0], &samplesSingle);
    for (int j = 0; j < k; ++j) {
      REQUIRE(nns[i*k + j] == nnSingle[j]);
      REQUIRE(dists[i*k + j] == Approx(distsSingle[j]).epsilon(EPSILON));
    }
    REQUIRE(samplesSingle == nX);

    set<int> nnSet(nnSingle.begin(), nnSingle.end());
    int maxNnIdx = nnSingle[k-1];
    float maxNnDist = array::euclideanDistance(d, q, X + maxNnIdx*d, &scratch[0]);
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = array::euclideanDistance(d, q, x, &scratch[0]);
      if (dist == maxNnDist)
	REQUIRE(j == maxNnIdx);
      else if (dist < maxNnDist)
	REQUIRE(nnSet.count(j));
      else
	REQUIRE(!nnSet.count(j));
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan4", "[linearscan]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  vector<double> XSqNorms(n);
  array::sqNorm(n, d, data, &XSqNorms[0]);
  vector<double> allDists(n*n);
  vector<double> scratch(std::max(d, 2*n));
  array::euclideanSqDistances(n, n, d, data, data, &XSqNorms[0], &allDists[0], &scratch[0]);
  
  LinearScan<double> ls(Metric::EUCLIDEAN);
  REQUIRE_THROWS_AS(ls.fit(0,d,data), std::invalid_argument);
  REQUIRE_THROWS_AS(ls.fit(n,0,data), std::invalid_argument);
  ls.fit(n,d,data);
  vector<int> nns(1);
  vector<double> dists(1);
  vector<int> samples(1);
  ls.query(1, d, 1, data, &nns[0], &dists[0], &samples[0]);
  REQUIRE(nns[0] == 0);
  REQUIRE((dists[0] == 0 || dists[0] == std::sqrt(allDists[0])));
  REQUIRE(samples[0] == n);

  
  int m = 100;
  int k = 50;
  nns = vector<int>(m*k);
  dists = vector<double>(m*k);
  samples = vector<int>(m);
  ls.query(m, d, k, data, &nns[0], &dists[0], &samples[0]);
  for (int i = 0; i < m; ++i) {
    REQUIRE(nns[i*k] == i);
    REQUIRE(dists[i*k] >= 0);
    REQUIRE((dists[i*k] == 0 || dists[i*k] == std::sqrt(allDists[i*n+i])));
    REQUIRE(samples[i] == n);
  }

  m = 5;
  k = n + 2;
  nns = vector<int>(m*k);
  dists = vector<double>(m*k);
  ls.query(m, d, k, data, &nns[0], &dists[0], &samples[0]);
  for (int i = 0; i < m; ++i) {
    REQUIRE(nns[i*k] == i);
    REQUIRE(nns[i*k + n] == -1);
    REQUIRE(nns[i*k + n + 1] == -1);
    REQUIRE((dists[i*k] == 0 || dists[i*k] == std::sqrt(allDists[i*n + i])));
    REQUIRE(dists[i*k + n] == -1);
    REQUIRE(dists[i*k + n + 1] == -1);
    REQUIRE(samples[i] == n);
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan5", "[linearscan]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  
  LinearScan<double> ls(Metric::EUCLIDEAN);
  ls.fit(nX, d, X);

  const double EPSILON = 1e-12;
  
  int k = 1;
  vector<int> nns(nY*k);
  vector<double> dists(nY*k);
  vector<double> scratch(d);
  vector<int> samples(nY);

  ls.query(nY, d, k, Y, &nns[0], &dists[0], &samples[0]);
  for (int s : samples)
    REQUIRE(s == nX);

  for (int i = 0; i < nY; ++i) {
    double* q = Y + i*d;
    int nnIdx;
    double nnDist = 99999999999999;
    for (int j = 0; j < nX; ++j) {
      double* x = X + j*d;
      double dist = array::euclideanDistance(d, q, x, &scratch[0]);
      if (dist < nnDist) {
	nnDist = dist;
	nnIdx = j;
      }
    }

    int nn;
    double dist;
    int samp;
    ls.query(1, d, 1, q, &nn, &dist, &samp);
    REQUIRE(nns[i] == nn);
    REQUIRE(nn == nnIdx);
    REQUIRE(dists[i] == Approx(dist).epsilon(EPSILON));
    REQUIRE(dist == Approx(nnDist).epsilon(EPSILON));
    REQUIRE(samp == nX);
    REQUIRE(samp == samples[i]);
    for (int j = 0; j < nX; ++j) {
      double* x = X + j*d;
      double dist = array::euclideanDistance(d, q, x, &scratch[0]);
      if (j != nn)
	REQUIRE(dist > nnDist);
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan6", "[linearscan]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const double EPSILON = 1e-12;
  
  LinearScan<double> ls(Metric::EUCLIDEAN);
  ls.fit(nX, d, X);

  int k = 50;
  vector<int> nns(nY*k);
  vector<double> dists(nY*k);
  vector<double> scratch(d);
  vector<int> samples(nY);

  ls.query(nY, d, k, Y, &nns[0], &dists[0], &samples[0]);
  for (int s : samples)
    REQUIRE(s == nX);

  vector<int> nnSingle(k);
  vector<double> distsSingle(k);
  int samplesSingle;
  for (int i = 0; i < nY; ++i) {
    double* q = Y + i*d;
    ls.query(1, d, k, q, &nnSingle[0], &distsSingle[0], &samplesSingle);
    for (int j = 0; j < k; ++j) {
      REQUIRE(nns[i*k + j] == nnSingle[j]);
      REQUIRE(dists[i*k + j] == Approx(distsSingle[j]).epsilon(EPSILON));
    }
    REQUIRE(samplesSingle == nX);

    set<int> nnSet(nnSingle.begin(), nnSingle.end());
    int maxNnIdx = nnSingle[k-1];
    double maxNnDist = array::euclideanDistance(d, q, X + maxNnIdx*d, &scratch[0]);
    for (int j = 0; j < nX; ++j) {
      double* x = X + j*d;
      double dist = array::euclideanDistance(d, q, x, &scratch[0]);
      if (dist == maxNnDist)
	REQUIRE(j == maxNnIdx);
      else if (dist < maxNnDist)
	REQUIRE(nnSet.count(j));
      else
	REQUIRE(!nnSet.count(j));
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan7", "[linearscan]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  
  LinearScan<float> ls(Metric::TAXICAB);
  REQUIRE_THROWS_AS(ls.fit(0,d,data), std::invalid_argument);
  REQUIRE_THROWS_AS(ls.fit(n,0,data), std::invalid_argument);
  ls.fit(n,d,data);
  vector<int> nns(1);
  vector<float> dists(1);
  vector<int> samples(1);
  ls.query(1, d, 1, data, &nns[0], &dists[0], &samples[0]);
  REQUIRE(nns[0] == 0);
  REQUIRE(dists[0] == 0);
  REQUIRE(samples[0] == n);

  int m = 100;
  int k = 50;
  nns = vector<int>(m*k);
  dists = vector<float>(m*k);
  samples = vector<int>(m);
  ls.query(m, d, k, data, &nns[0], &dists[0], &samples[0]);
  for (int i = 0; i < m; ++i) {
    REQUIRE(nns[i*k] == i);
    REQUIRE(dists[i*k] == 0);
    REQUIRE(samples[i] == n);
  }

  m = 5;
  k = n + 2;
  nns = vector<int>(m*k);
  dists = vector<float>(m*k);
  samples = vector<int>(m);
  ls.query(m, d, k, data, &nns[0], &dists[0], &samples[0]);
  for (int i = 0; i < m; ++i) {
    REQUIRE(nns[i*k] == i);
    REQUIRE(nns[i*k + n] == -1);
    REQUIRE(nns[i*k + n + 1] == -1);
    REQUIRE(dists[i*k] == 0);
    REQUIRE(dists[i*k + n] == -1);
    REQUIRE(dists[i*k + n + 1] == -1);
    REQUIRE(samples[i] == n);
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan8", "[linearscan]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  
  LinearScan<float> ls(Metric::TAXICAB);
  ls.fit(nX, d, X);

  int k = 1;
  vector<int> nns(nY*k);
  vector<float> dists(nY*k);
  vector<float> scratch(d);
  vector<int> samples(nY);

  ls.query(nY, d, k, Y, &nns[0], &dists[0], &samples[0]);
  for (int s : samples)
    REQUIRE(s == nX);

  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    int nnIdx;
    float nnDist = 99999999999999;
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = array::taxicabDistance(d, q, x, &scratch[0]);
      if (dist < nnDist) {
	nnDist = dist;
	nnIdx = j;
      }
    }

    int nn;
    float dist;
    int samp;
    ls.query(1, d, 1, q, &nn, &dist, &samp);
    REQUIRE(nns[i] == nn);
    REQUIRE(nn == nnIdx);
    REQUIRE(dists[i] == dist);
    REQUIRE(dist == nnDist);
    REQUIRE(samp == samples[i]);
    REQUIRE(samp == nX);
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = array::taxicabDistance(d, q, x, &scratch[0]);
      if (j != nn)
	REQUIRE(dist > nnDist);
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan9", "[linearscan]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  
  LinearScan<float> ls(Metric::TAXICAB);
  ls.fit(nX, d, X);

  int k = 50;
  vector<int> nns(nY*k);
  vector<float> scratch(d);
  vector<float> dists(nY*k);
  vector<int> samples(nY);

  ls.query(nY, d, k, Y, &nns[0], &dists[0], &samples[0]);

  vector<int> nnSingle(k);
  vector<float> distsSingle(k);
  int samplesSingle;
  for (int i = 0; i < nY; ++i) {
    float* q = Y + i*d;
    ls.query(1, d, k, q, &nnSingle[0], &distsSingle[0], &samplesSingle);
    for (int j = 0; j < k; ++j) {
      REQUIRE(nns[i*k + j] == nnSingle[j]);
      REQUIRE(dists[i*k + j] == distsSingle[j]);
    }
    REQUIRE(samples[i] == nX);
    REQUIRE(samples[i] == samplesSingle);

    set<int> nnSet(nnSingle.begin(), nnSingle.end());
    int maxNnIdx = nnSingle[k-1];
    float maxNnDist = array::taxicabDistance(d, q, X + maxNnIdx*d, &scratch[0]);
    for (int j = 0; j < nX; ++j) {
      float* x = X + j*d;
      float dist = array::taxicabDistance(d, q, x, &scratch[0]);
      if (dist == maxNnDist)
	REQUIRE(j == maxNnIdx);
      else if (dist < maxNnDist)
	REQUIRE(nnSet.count(j));
      else
	REQUIRE(!nnSet.count(j));
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan10", "[linearscan]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  
  LinearScan<double> ls(Metric::TAXICAB);
  REQUIRE_THROWS_AS(ls.fit(0,d,data), std::invalid_argument);
  REQUIRE_THROWS_AS(ls.fit(n,0,data), std::invalid_argument);
  ls.fit(n,d,data);
  vector<int> nns(1);
  vector<double> dists(1);
  vector<int> samples(1);
  ls.query(1, d, 1, data, &nns[0], &dists[0], &samples[0]);
  REQUIRE(nns[0] == 0);
  REQUIRE(dists[0] == 0);
  REQUIRE(samples[0] == n);

  int m = 100;
  int k = 50;
  nns = vector<int>(m*k);
  dists = vector<double>(m*k);
  samples = vector<int>(m);
  ls.query(m, d, k, data, &nns[0], &dists[0], &samples[0]);
  for (int i = 0; i < m; ++i) {
    REQUIRE(nns[i*k] == i);
    REQUIRE(dists[i*k] == 0);
    REQUIRE(samples[i] == n);
  }

  m = 5;
  k = n + 2;
  nns = vector<int>(m*k);
  dists = vector<double>(m*k);
  samples = vector<int>(m);
  ls.query(m, d, k, data, &nns[0], &dists[0], &samples[0]);
  for (int i = 0; i < m; ++i) {
    REQUIRE(nns[i*k] == i);
    REQUIRE(nns[i*k + n] == -1);
    REQUIRE(nns[i*k + n + 1] == -1);
    REQUIRE(dists[i*k] == 0);
    REQUIRE(dists[i*k + n] == -1);
    REQUIRE(dists[i*k + n + 1] == -1);
    REQUIRE(samples[i] == n);
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan11", "[linearscan]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  
  LinearScan<double> ls(Metric::TAXICAB);
  ls.fit(nX, d, X);

  int k = 1;
  vector<int> nns(nY*k);
  vector<double> dists(nY*k);
  vector<double> scratch(d);
  vector<int> samples(nY);

  ls.query(nY, d, k, Y, &nns[0], &dists[0], &samples[0]);
  for (int s : samples)
    REQUIRE(s == nX);

  for (int i = 0; i < nY; ++i) {
    double* q = Y + i*d;
    int nnIdx;
    double nnDist = 99999999999999;
    for (int j = 0; j < nX; ++j) {
      double* x = X + j*d;
      double dist = array::taxicabDistance(d, q, x, &scratch[0]);
      if (dist < nnDist) {
	nnDist = dist;
	nnIdx = j;
      }
    }

    int nn;
    double dist;
    int samp;
    ls.query(1, d, 1, q, &nn, &dist, &samp);
    REQUIRE(nns[i] == nn);
    REQUIRE(nn == nnIdx);
    REQUIRE(dists[i] == dist);
    REQUIRE(dist == nnDist);
    REQUIRE(samp == samples[i]);
    REQUIRE(samp == nX);
    for (int j = 0; j < nX; ++j) {
      double* x = X + j*d;
      double dist = array::taxicabDistance(d, q, x, &scratch[0]);
      if (j != nn)
	REQUIRE(dist > nnDist);
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan12", "[linearscan]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  
  LinearScan<double> ls(Metric::TAXICAB);
  ls.fit(nX, d, X);

  int k = 50;
  vector<int> nns(nY*k);
  vector<double> dists(nY*k);
  vector<double> scratch(d);
  vector<int> samples(nY);

  ls.query(nY, d, k, Y, &nns[0], &dists[0], &samples[0]);

  vector<int> nnSingle(k);
  vector<double> distsSingle(k);
  int samplesSingle;
  for (int i = 0; i < nY; ++i) {
    double* q = Y + i*d;
    ls.query(1, d, k, q, &nnSingle[0], &distsSingle[0], &samplesSingle);
    for (int j = 0; j < k; ++j) {
      REQUIRE(nns[i*k + j] == nnSingle[j]);
      REQUIRE(dists[i*k + j] == distsSingle[j]);
    }
    REQUIRE(samples[i] == nX);
    REQUIRE(samplesSingle == samples[i]);

    set<int> nnSet(nnSingle.begin(), nnSingle.end());
    int maxNnIdx = nnSingle[k-1];
    double maxNnDist = array::taxicabDistance(d, q, X + maxNnIdx*d, &scratch[0]);
    for (int j = 0; j < nX; ++j) {
      double* x = X + j*d;
      double dist = array::taxicabDistance(d, q, x, &scratch[0]);
      if (dist == maxNnDist)
	REQUIRE(j == maxNnIdx);
      else if (dist < maxNnDist)
	REQUIRE(nnSet.count(j));
      else
	REQUIRE(!nnSet.count(j));
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_linear_scan13", "[linearscan]" ) {
  float* data1;
  int n, d;
  constructMediumTestSet(&data1, &n, &d);
  float *X1, *Y1;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data1, &X1, &Y1);

  double* data2;
  constructMediumTestSet(&data2, &n, &d);
  double *X2, *Y2;
  array::splitInHalf(n, d, &nX, &nY, data2, &X2, &Y2);

  for (Metric metric : { Metric::EUCLIDEAN, Metric::TAXICAB }) {
    LinearScan<float> ls1(metric);
    LinearScan<double> ls2(metric);
    ls1.fit(nX, d, X1);
    ls2.fit(nY, d, Y2);

    vector<int> nns1(nX*nY);
    vector<int> nns2(nX*nY);
    vector<float> dists1(nX*nY);
    vector<double> dists2(nX*nY);
    vector<float> scratch1(d);
    vector<double> scratch2(d);
    vector<int> samples1(nY);
    vector<int> samples2(nX);

    ls1.query(nY, d, nX, Y1, &nns1[0], &dists1[0], &samples1[0]);
    ls2.query(nX, d, nY, X2, &nns2[0], &dists2[0], &samples2[0]);

    for (int s : samples1)
      REQUIRE(s == nX);
    for (int s : samples2)
      REQUIRE(s == nY);

    for (int i = 0; i < nY; ++i) {
      set<int> nns(nns1.begin() + i*nX, nns1.begin() + (i+1)*nX);
      for (int j = 0; j < nX; ++j)
	REQUIRE(nns.count(j));
    }

    for (int i = 0; i < nX; ++i) {
      set<int> nns(nns2.begin() + i*nY, nns2.begin() + (i+1)*nY);
      for (int j = 0; j < nY; ++j)
	REQUIRE(nns.count(j));
    }

    const float EPSILON1 = 1e-3f;
    const double EPSILON2 = 1e-12;

    for (int i = 0; i < nY; ++i) {
      float* y = Y1 + i*d;
      for (int j = 0; j < nX; ++j) {
	float* x = X1 + nns1[i*nX + j]*d;
	
	if (metric == Metric::EUCLIDEAN) {
	  float dist = array::euclideanDistance(d, y, x, &scratch1[0]);
	  REQUIRE(dists1[i*nX + j] == Approx(dist).epsilon(EPSILON1));
	}
	if (metric == Metric::TAXICAB) {
	  float dist = array::taxicabDistance(d, y, x, &scratch1[0]);
	  REQUIRE(dists1[i*nX + j] == dist);
	}
      }
    }

    for (int i = 0; i < nX; ++i) {
      double* x = X2 + i*d;
      for (int j = 0; j < nY; ++j) {
	double* y = Y2 + nns2[i*nY + j]*d;
	if (metric == Metric::EUCLIDEAN) {
	  double dist = array::euclideanDistance(d, y, x, &scratch2[0]);
	  REQUIRE(dists2[i*nY + j] == Approx(dist).epsilon(EPSILON2));
	}
	if (metric == Metric::TAXICAB) {
	  double dist = array::taxicabDistance(d, y, x, &scratch2[0]);
	  REQUIRE(dists2[i*nY + j] == dist);
	}
      }
    }
  }
  allocator.free(data1);
  allocator.free(data2);
}



TEST_CASE( "test_linear_scan14", "[linearscan]" ) {
  float* data1;
  int n, d;
  constructMediumTestSet(&data1, &n, &d);
  float *X1, *Y1;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data1, &X1, &Y1);

  double* data2;
  constructMediumTestSet(&data2, &n, &d);
  double *X2, *Y2;
  array::splitInHalf(n, d, &nX, &nY, data2, &X2, &Y2);

  for (Metric metric : { Metric::EUCLIDEAN, Metric::TAXICAB }) {
    LinearScan<float> ls1(metric, true, true);
    LinearScan<float> ls2(metric, false, false);
    LinearScan<double> ls3(metric, true, true);
    LinearScan<double> ls4(metric, false, false);
    ls1.fit(nX, d, X1);
    ls2.fit(nX, d, X1);   
    ls3.fit(nX, d, X2);
    ls4.fit(nX, d, X2);   

    vector<int> nns1(nX*nY);
    vector<int> nns2(nX*nY);
    vector<int> nns3(nX*nY);
    vector<int> nns4(nX*nY);
    vector<float> dists1(nX*nY);
    vector<float> dists2(nX*nY);
    vector<double> dists3(nX*nY);
    vector<double> dists4(nX*nY);
    vector<int> samples1(nY);
    vector<int> samples2(nY);
    vector<int> samples3(nY);
    vector<int> samples4(nY);
    vector<float> scratch1(d);
    vector<double> scratch2(d);

    ls1.query(nY, d, nX, Y1, &nns1[0], &dists1[0], &samples1[0]);
    ls2.query(nY, d, nX, Y1, &nns2[0], &dists2[0], &samples2[0]);
    ls3.query(nY, d, nX, Y2, &nns3[0], &dists3[0], &samples3[0]);
    ls4.query(nY, d, nX, Y2, &nns4[0], &dists4[0], &samples4[0]);

    REQUIRE(array::allEqual(nY, &samples1[0], nX));
    REQUIRE(array::allEqual(nY, &samples2[0], -1));
    REQUIRE(array::allEqual(nY, &samples3[0], nX));
    REQUIRE(array::allEqual(nY, &samples4[0], -1));

    const float EPSILON1 = 1e-3f;
    const double EPSILON2 = 1e-12;

    REQUIRE(array::equal(nY, &nns1[0], &nns2[0]));
    REQUIRE(array::equal(nY, &nns3[0], &nns4[0]));
    
    for (int i = 0; i < nY; ++i) {
      float* y1 = Y1 + i*d;
      double* y2 = Y2 + i*d;
      for (int j = 0; j < nX; ++j) {
	float* x1 = X1 + nns1[i*nX + j]*d;
	double* x2 = X2 + nns3[i*nX + j]*d;
	
	if (metric == Metric::EUCLIDEAN) {
	  float dist1 = array::euclideanDistance(d, y1, x1, &scratch1[0]);
	  REQUIRE(dists1[i*nX + j] == Approx(dist1).epsilon(EPSILON1));
	  double dist2 = array::euclideanDistance(d, y2, x2, &scratch2[0]);
	  REQUIRE(dists3[i*nX + j] == Approx(dist2).epsilon(EPSILON2));
	}
	if (metric == Metric::TAXICAB) {
	  float dist1 = array::taxicabDistance(d, y1, x1, &scratch1[0]);
	  REQUIRE(dists1[i*nX + j] == dist1);
	  double dist2 = array::taxicabDistance(d, y2, x2, &scratch2[0]);
	  REQUIRE(dists3[i*nX + j] == dist2);
	}
      }
    }

    REQUIRE(array::allEqual(nY, &dists2[0], -1.0f));
    REQUIRE(array::allEqual(nY, &dists4[0], -1.0));
  }
  allocator.free(data1);
  allocator.free(data2);
}



TEST_CASE( "test_random_sampling1", "[randomsampling]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  uint32_t seed = 9371;

  float h = 28;

  vector<float> scratch(d); 
  vector<float> mu(nY);
  vector<float> Z1(nY);
  vector<float> Z2(nY);
  vector<int> samples(nY);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN, Kernel::LAPLACIAN }) {
    for (int m  : { 1, 5, 123, nX, 2*nX }) {
      seed = mt19937(seed)();
      KdeEstimator::PseudoRandomNumberGenerator rng(nX, seed);
      
      REQUIRE_THROWS_AS(RandomSampling<float>(0, kernel, m), std::invalid_argument);
      REQUIRE_THROWS_AS(RandomSampling<float>(-1, kernel, m), std::invalid_argument);
      REQUIRE_THROWS_AS(RandomSampling(h, kernel, -1), std::invalid_argument);
      REQUIRE_THROWS_AS(RandomSampling(h, kernel, 0), std::invalid_argument);
      RandomSampling rs1(h, kernel, m, seed);
      REQUIRE_THROWS_AS(rs1.query(d, Y, &mu[0], &samples[0]), std::invalid_argument);
      REQUIRE_THROWS_AS(rs1.fit(0, d, X), std::invalid_argument);
      REQUIRE_THROWS_AS(rs1.fit(nX, 0, X), std::invalid_argument);
      rs1.fit(nX, d, X);

      REQUIRE_THROWS_AS(rs1.query(d-1, X, &mu[0], &samples[0]), std::invalid_argument);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      uint32_t rn1 = rng();
      REQUIRE(rn1 == rs1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS      
      
      for (int i = 0; i < nY; ++i) {
	float* q = Y + i*d;
	mu[i] = 0;
	for (int j = 0; j < m; ++j) {
	  uint32_t idx = rng();
	  float* x = X + idx*d;
	  float dist;
	  if (kernel == Kernel::LAPLACIAN)
	    dist = array::taxicabDistance(d, q, x, &scratch[0]);
	  else
	    dist = array::euclideanDistance(d, q, x, &scratch[0]);
	  if (kernel == Kernel::GAUSSIAN)
	    mu[i] += std::exp(-dist*dist/h/h/2);
	  else
	    mu[i] += std::exp(-dist/h);
	}
	mu[i] /= m;
      }
  
  
      for (int i = 0; i < nY; ++i) {
	float* q = Y + i*d;
        rs1.query(d, q, &Z1[i], &samples[i]);
      }
      for (int s : samples)
        REQUIRE(s == m);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS      
      uint32_t rn2 = rng();
      REQUIRE(rn2 == rs1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS      

      rs1.resetSeed(seed);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS      
      REQUIRE(rn1 == rs1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS      
      rs1.query(nY, d, Y, &Z2[0], &samples[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS      
      REQUIRE(rn2 == rs1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS      

      for (int s : samples)
        REQUIRE(s == m);

      for (int i = 0; i < nY; ++i) {
	float absError, relError;
	absError = std::abs(Z1[i]-mu[i]);
	relError = std::abs(Z1[i]-mu[i])/mu[i];
	REQUIRE(absError < 1e-7);
	REQUIRE(relError < 1e-5);
	absError = std::abs(Z2[i]-mu[i]);
	relError = std::abs(Z2[i]-mu[i])/mu[i];
	REQUIRE(absError < 1e-7);
	REQUIRE(relError < 1e-5);
	REQUIRE(Z1[i] == Z2[i]);
      }
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_random_sampling2", "[randomsampling]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  uint32_t seed = 3917;

  double h = 29;

  vector<double> scratch(d); 
  vector<double> mu(nY);
  vector<double> Z1(nY);
  vector<double> Z2(nY);
  vector<int> samples(nY);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN, Kernel::LAPLACIAN }) {
    for (int m  : { 1, 7, 234, nX, 3*nX+1 }) {
      seed = mt19937(seed)();
      KdeEstimator::PseudoRandomNumberGenerator rng(nX, seed);
      
      REQUIRE_THROWS_AS(RandomSampling<double>(0, kernel, m), std::invalid_argument);
      REQUIRE_THROWS_AS(RandomSampling<double>(-1, kernel, m), std::invalid_argument);
      REQUIRE_THROWS_AS(RandomSampling(h, kernel, -1), std::invalid_argument);
      REQUIRE_THROWS_AS(RandomSampling(h, kernel, 0), std::invalid_argument);
      RandomSampling rs1(h, kernel, m, seed);
      REQUIRE_THROWS_AS(rs1.query(d, Y, &mu[0], &samples[0]), std::invalid_argument);
      REQUIRE_THROWS_AS(rs1.fit(0, d, X), std::invalid_argument);
      REQUIRE_THROWS_AS(rs1.fit(nX, 0, X), std::invalid_argument);
      rs1.fit(nX, d, X);

      REQUIRE_THROWS_AS(rs1.query(d-1, X, &mu[0], &samples[0]), std::invalid_argument);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS      
      uint32_t rn1 = rng();
      REQUIRE(rn1 == rs1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS      

      for (int i = 0; i < nY; ++i) {
	double* q = Y + i*d;
	mu[i] = 0;
	for (int j = 0; j < m; ++j) {
	  uint32_t idx = rng();
	  double* x = X + idx*d;
	  double dist;
	  if (kernel == Kernel::LAPLACIAN)
	    dist = array::taxicabDistance(d, q, x, &scratch[0]);
	  else
	    dist = array::euclideanDistance(d, q, x, &scratch[0]);
	  if (kernel == Kernel::GAUSSIAN)
	    mu[i] += std::exp(-dist*dist/h/h/2);
	  else
	    mu[i] += std::exp(-dist/h);
	}
	mu[i] /= m;
      }
  
  
      for (int i = 0; i < nY; ++i) {
	double* q = Y + i*d;
	rs1.query(d, q, &Z1[i], &samples[i]);
      }
      for (int s : samples)
        REQUIRE(s == m);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS      
      uint32_t rn2 = rng();
      REQUIRE(rn2 == rs1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS      

      rs1.resetSeed(seed);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS      
      REQUIRE(rn1 == rs1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      rs1.query(nY, d, Y, &Z2[0], &samples[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(rn2 == rs1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      
      for (int s : samples)
        REQUIRE(s == m);
      
      for (int i = 0; i < nY; ++i) {
	float absError, relError;
	absError = std::abs(Z1[i]-mu[i]);
	relError = std::abs(Z1[i]-mu[i])/mu[i];
	REQUIRE(absError < 1e-15);
	REQUIRE(relError < 1e-13);
	absError = std::abs(Z2[i]-mu[i]);
	relError = std::abs(Z2[i]-mu[i])/mu[i];
	REQUIRE(absError < 1e-15);
	REQUIRE(relError < 1e-13);
	REQUIRE(Z1[i] == Z2[i]);
      }
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_random_sampling3", "[randomsampling]" ) {
  const double h = 16;
  float* data1;
  double* data2;
  int n, d;
  constructMediumTestSet(&data1, &n, &d);

  float *X1, *Y1;
  double *X2, *Y2;
  int nX, nY;
  constructMediumTestSet(&data2, &n, &d);
  array::splitInHalf(n,d,&nX,&nY,data1,&X1,&Y1);
  array::splitInHalf(n,d,&nX,&nY,data2,&X2,&Y2);

  vector<int> samples(nY);

  const uint32_t seed = 918273;

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN, Kernel::LAPLACIAN }) {

    RandomSampling<float> rs1(h, kernel, 123);
    RandomSampling<double> rs2(h, kernel, 123);

    REQUIRE_NOTHROW(rs1.resetParameters());
    REQUIRE_NOTHROW(rs1.resetParameters(1));
    REQUIRE_THROWS_AS(rs1.resetParameters(0), invalid_argument);
    REQUIRE_THROWS_AS(rs1.resetParameters(-1), invalid_argument);
    REQUIRE_NOTHROW(rs1.resetParameters(nullopt));
    REQUIRE_THROWS_AS(rs1.resetParameters(nullopt,1), invalid_argument);
    REQUIRE_THROWS_AS(rs1.resetParameters(1,2), invalid_argument);
    REQUIRE_NOTHROW(rs1.resetParameters(nullopt,nullopt));
    REQUIRE_NOTHROW(rs1.resetParameters(1,nullopt));
    REQUIRE_THROWS_AS(rs1.resetParameters(0,nullopt), invalid_argument);
    REQUIRE_THROWS_AS(rs1.resetParameters(-1,nullopt), invalid_argument);
  
    REQUIRE_NOTHROW(rs2.resetParameters());
    REQUIRE_NOTHROW(rs2.resetParameters(1));
    REQUIRE_THROWS_AS(rs2.resetParameters(0), invalid_argument);
    REQUIRE_THROWS_AS(rs2.resetParameters(-1), invalid_argument);
    REQUIRE_NOTHROW(rs2.resetParameters(nullopt));
    REQUIRE_THROWS_AS(rs2.resetParameters(nullopt,1), invalid_argument);
    REQUIRE_THROWS_AS(rs2.resetParameters(1,2), invalid_argument);
    REQUIRE_NOTHROW(rs2.resetParameters(nullopt,nullopt));
    REQUIRE_NOTHROW(rs2.resetParameters(1,nullopt));
    REQUIRE_THROWS_AS(rs2.resetParameters(0,nullopt), invalid_argument);
    REQUIRE_THROWS_AS(rs2.resetParameters(-1,nullopt), invalid_argument);

    int m1 = 123;
    int m2 = 234;
    int m3 = 345;
  
    rs1 = RandomSampling<float>(h, kernel, m1);
    rs2 = RandomSampling<double>(h, kernel, m1);
    rs1.fit(nX, d, X1);
    rs2.fit(nX, d, X2);
    vector<float> Z1_1(nY);
    vector<double> Z2_1(nY);
    vector<float> Z1_2(nY);
    vector<double> Z2_2(nY);

    for (int m : { m1, m2, m3 }) {
      rs1.resetSeed(seed);
      rs1.resetParameters(m);
      rs1.query(nY, d, Y1, &Z1_1[0], &samples[0]);

      for (int s : samples)
        REQUIRE(s == m);

      rs2.resetSeed(seed);
      rs2.resetParameters(m);
      rs2.query(nY, d, Y2, &Z2_1[0], &samples[0]);
      for (int s : samples)
        REQUIRE(s == m);

      RandomSampling<float> rs3(h, kernel, m, seed);
      rs3.fit(nX, d, X1);
      rs3.query(nY, d, Y1, &Z1_2[0], &samples[0]);
      for (int s : samples)
        REQUIRE(s == m);      

      REQUIRE(array::equal(nY, &Z1_1[0], &Z1_2[0]));

      RandomSampling<double> rs4(h, kernel, m, seed);
      rs4.fit(nX, d, X2);
      rs4.query(nY, d, Y2, &Z2_2[0], &samples[0]);    
      for (int s : samples)
        REQUIRE(s == m);      
      REQUIRE(array::equal(nY, &Z2_1[0], &Z2_2[0]));
    }
  }
  
  allocator.free(data1);
  allocator.free(data2);
}



TEST_CASE( "test_kde_subset1", "[kdesubset]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const uint32_t seed = 0xb1c72b79;
  const int k = 300;
  const float h = 23;
  const float EPSILON = 1e-4f;
  const float DELTA = 1e-4f;
  
  mt19937 rng(seed);
  uniform_int_distribution<uint32_t> ud(0,nX-1);
  
  vector<float> scratch(k+d);
  vector<float> dists(k);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN, Kernel::LAPLACIAN }) {
    vector<float> mu(nY);
    vector<float> Z(nY);
    vector<float> Z2(nY);
    for (int i = 0; i < nY; ++i) {
      float* y = Y + i*d;
      vector<uint32_t> subset(k);
      for (int j = 0; j < k; ++j)
	subset[j] = ud(rng);

      mu[i] = 0;
      for (int j = 0; j < k; ++j) {
	float* x = X + subset[j]*d;
	float dist;
	if (kernel == Kernel::LAPLACIAN)
	  dist = array::taxicabDistance(d, y, x, &scratch[0]);
	else
	  dist = array::euclideanDistance(d, y, x, &scratch[0]);
	dists[j] = dist;
	if (kernel == Kernel::GAUSSIAN)
	  mu[i] += std::exp(-dist*dist/h/h/2);
	else
	  mu[i] += std::exp(-dist/h);
      }
      mu[i] /= k;

      Z[i] = kdeSubset(k, d, h, X, y, &subset[0], &scratch[0], kernel);
      REQUIRE(mu[i] == Approx(Z[i]).epsilon(EPSILON));
      REQUIRE(mu[i] == Approx(Z[i]).epsilon(0).margin(DELTA));
      REQUIRE(std::abs(mu[i]-Z[i]) < DELTA);

      Z2[i] = kdeDists(k, h, &dists[0], &scratch[0], kernel);
      REQUIRE(mu[i] == Approx(Z2[i]).epsilon(EPSILON));
      REQUIRE(mu[i] == Approx(Z2[i]).epsilon(0).margin(DELTA));
      REQUIRE(std::abs(mu[i]-Z2[i]) < DELTA);
    }

    for (int i = 0; i < nY; ++i) {
      REQUIRE(mu[i] == Approx(Z[i]).epsilon(EPSILON));
      REQUIRE(mu[i] == Approx(Z[i]).epsilon(0).margin(DELTA));
      REQUIRE(std::abs(mu[i]-Z[i]) < DELTA);
      REQUIRE(mu[i] == Approx(Z2[i]).epsilon(EPSILON));
      REQUIRE(mu[i] == Approx(Z2[i]).epsilon(0).margin(DELTA));
      REQUIRE(std::abs(mu[i]-Z2[i]) < DELTA);
    }

    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));
  }
  allocator.free(data);
}



TEST_CASE( "test_kde_subset2", "[kdesubset]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const uint32_t seed = 0xb30a7eaf;
  const int k = 300;
  const double h = 23;
  const double EPSILON = 1e-14;
  const double DELTA = 1e-16;
  
  mt19937 rng(seed);
  uniform_int_distribution<uint32_t> ud(0,nX-1);
  
  vector<double> scratch(k+d);
  vector<double> dists(k);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN, Kernel::LAPLACIAN }) {
    vector<double> mu(nY);
    vector<double> Z(nY);
    vector<double> Z2(nY);
    for (int i = 0; i < nY; ++i) {
      double* y = Y + i*d;
      vector<uint32_t> subset(k);
      for (int j = 0; j < k; ++j)
	subset[j] = ud(rng);

      mu[i] = 0;
      for (int j = 0; j < k; ++j) {
	double* x = X + subset[j]*d;
	double dist;
	if (kernel == Kernel::LAPLACIAN)
	  dist = array::taxicabDistance(d, y, x, &scratch[0]);
	else
	  dist = array::euclideanDistance(d, y, x, &scratch[0]);
	dists[j] = dist;
	if (kernel == Kernel::GAUSSIAN)
	  mu[i] += std::exp(-dist*dist/h/h/2);
	else
	  mu[i] += std::exp(-dist/h);
      }
      mu[i] /= k;

      Z[i] = kdeSubset(k, d, h, X, y, &subset[0], &scratch[0], kernel);
      REQUIRE(mu[i] == Approx(Z[i]).epsilon(EPSILON));
      REQUIRE(mu[i] == Approx(Z[i]).epsilon(0).margin(DELTA));
      REQUIRE(std::abs(mu[i]-Z[i]) < DELTA);

      Z2[i] = kdeDists(k, h, &dists[0], &scratch[0], kernel);
      REQUIRE(mu[i] == Approx(Z2[i]).epsilon(EPSILON));
      REQUIRE(mu[i] == Approx(Z2[i]).epsilon(0).margin(DELTA));
      REQUIRE(std::abs(mu[i]-Z2[i]) < DELTA);
    }

    for (int i = 0; i < nY; ++i) {
      REQUIRE(mu[i] == Approx(Z[i]).epsilon(EPSILON));
      REQUIRE(mu[i] == Approx(Z[i]).epsilon(0).margin(DELTA));
      REQUIRE(std::abs(mu[i]-Z[i]) < DELTA);

      REQUIRE(mu[i] == Approx(Z2[i]).epsilon(EPSILON));
      REQUIRE(mu[i] == Approx(Z2[i]).epsilon(0).margin(DELTA));
      REQUIRE(std::abs(mu[i]-Z2[i]) < DELTA);
    }

    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));
  }
  allocator.free(data);
}



TEST_CASE( "test_ann_estimator1", "[annestimator]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  vector<int> samples(nY);
  
  int m = 0;

  float h = 24;

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN,
	Kernel::LAPLACIAN }) {
    LinearScan<float> ls(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			 Metric::EUCLIDEAN);
    ls.fit(nX, d, X);
    AnnEstimator<float, LinearScan<float>> annEst(h, kernel, nX, m, &ls);
    annEst.fit(nX, d, X);

    NaiveKde<float> nkde(h, kernel);
    nkde.fit(nX, d, X);

    for (int i = 0; i < nY; ++i) {
      float* q = Y + i*d;
      float mu;
      nkde.query(d, q, &mu);
      float Z;
      annEst.query(d, q, &Z, &samples[i]);
      REQUIRE(Z == Approx(mu).epsilon(1e-4));
      REQUIRE(Z == Approx(mu).epsilon(0).margin(1e-6));
    }
    for (int s : samples)
      REQUIRE(s == nX);
    vector<float> mu(nY);
    nkde.query(nY, d, Y, &mu[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nX);
    vector<float> Z(nY);
    annEst.query(nY, d, Y, &Z[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nX);
    REQUIRE(!array::equal(nY, &mu[0], &Z[0]));
    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], 1e-6f));
  }
  allocator.free(data);
}



TEST_CASE( "test_ann_estimator2", "[annestimator]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  int m = 0;

  vector<int> samples(nX);
  double h = 25;

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN,
	Kernel::LAPLACIAN }) {
    LinearScan<double> ls(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			 Metric::EUCLIDEAN);
    ls.fit(nY, d, Y);
    AnnEstimator<double, LinearScan<double>> annEst(h, kernel, nY, m, &ls);
    annEst.fit(nY, d, Y);

    NaiveKde<double> nkde(h, kernel);
    nkde.fit(nY, d, Y);

    for (int i = 0; i < nX; ++i) {
      double* q = X + i*d;
      double mu;
      nkde.query(d, q, &mu);
      double Z;
      annEst.query(d, q, &Z, &samples[i]);
      REQUIRE(Z == Approx(mu).epsilon(1e-13));
      REQUIRE(Z == Approx(mu).epsilon(0).margin(1e-14));
    }
    for (int s : samples)
      REQUIRE(s == nY);

    vector<double> mu(nX);
    nkde.query(nX, d, X, &mu[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nY);
    vector<double> Z(nX);
    annEst.query(nX, d, X, &Z[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nY);
    REQUIRE(!array::equal(nX, &mu[0], &Z[0]));
    REQUIRE(array::almostEqual(nX, &mu[0], &Z[0], 1e-14));
  }
  allocator.free(data);
}



TEST_CASE( "test_ann_estimator3", "[annestimator]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const int seed = 68435;
  const int k = 0;

  float h = 26;
  vector<int> samples(nY);
  vector<float> scratch(d);
  vector<float> mu1(nY);
  vector<float> mu2(nY);
  vector<float> Z1(nY);
  vector<float> Z2(nY);
  
  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN,
	Kernel::LAPLACIAN }) {
    for (int m : { 1, 5, 100, 500, nX, 2*nX }) {
      KdeEstimator::PseudoRandomNumberGenerator rng(nX, seed);
      LinearScan<float> ls(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			   Metric::EUCLIDEAN);
      ls.fit(nX, d, X);

      REQUIRE_THROWS_AS((AnnEstimator<float, LinearScan<float>>(0,kernel,k,m,&ls)),
			std::invalid_argument);
      REQUIRE_THROWS_AS((AnnEstimator<float, LinearScan<float>>(-1,kernel,k,m,&ls)),
			std::invalid_argument);
      REQUIRE_THROWS_AS((AnnEstimator<float, LinearScan<float>>(h,kernel,-1,m,&ls)),
			std::invalid_argument);
      REQUIRE_THROWS_AS((AnnEstimator<float, LinearScan<float>>(h,kernel,k,-1,&ls)),
			std::invalid_argument);
    
      AnnEstimator<float, LinearScan<float>> annEst1(h, kernel, k, m, &ls, seed);
      annEst1.fit(nX, d, X);

      AnnEstimator<float, LinearScan<float>> annEst2(h, kernel, k, m, &ls, seed);
      annEst2.fit(nX, d, X);

      for (int i = 0; i < nY; ++i) {
	float* q = Y + i*d;
	mu1[i] = 0;
	for (int j = 0; j < m; ++j) {
	  uint32_t idx = rng();
	  float* x = X + idx*d;
	  float dist = (kernel == Kernel::LAPLACIAN ?
			array::taxicabDistance(d,q,x,&scratch[0]) :
			array::euclideanDistance(d,q,x,&scratch[0]));
	  mu1[i] += (kernel == Kernel::GAUSSIAN ? std::exp(-dist*dist/h/h/2) :
		     std::exp(-dist/h));
	}
	mu1[i] = mu1[i] * nX / nX / m;
        annEst1.query(d, q, &Z1[i], &samples[i]);
	REQUIRE(Z1[i] == Approx(mu1[i]).epsilon(1e-5));
	REQUIRE(std::abs(Z1[i] - mu1[i]) < 1e-7);
      }
      for (int s : samples)
        REQUIRE(s == m);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      uint32_t rv1 = rng();
      REQUIRE(rv1 == annEst1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      
      annEst2.query(nY, d, Y, &Z2[0], &samples[0]);
      for (int s : samples)
        REQUIRE(s == m);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(rv1 == annEst2.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      
      REQUIRE(array::almostEqual(nY, &mu1[0], &Z1[0], 1e-7f));
      REQUIRE(array::almostEqual(nY, &mu1[0], &Z2[0], 1e-7f));
      REQUIRE(array::equal(nY, &Z1[0], &Z2[0]));
      
      RandomSampling rs(h, kernel, m, seed);
      rs.fit(nX, d, X);
      rs.query(nY, d, Y, &mu2[0], &samples[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(rv1 == rs.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      for (int s : samples)
        REQUIRE(s == m);

      for (int i = 0; i < nY; ++i) {
	REQUIRE(Z1[i] == Approx(mu2[i]).epsilon(1e-5));
	REQUIRE(Z1[i] == Approx(mu2[i]).epsilon(0).margin(1e-7));
	REQUIRE(Z2[i] == Approx(mu2[i]).epsilon(1e-5));
	REQUIRE(Z2[i] == Approx(mu2[i]).epsilon(0).margin(1e-7));	
      }
      
      REQUIRE(array::almostEqual(nY, &mu2[0], &Z1[0], 1e-7f));
      REQUIRE(array::almostEqual(nY, &mu2[0], &Z2[0], 1e-7f));
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_ann_estimator4", "[annestimator]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const int seed = 79546;

  const int k = 0;

  double h = 29;
  vector<double> scratch(d);
  vector<double> mu1(nY);
  vector<double> mu2(nY);
  vector<double> Z1(nY);
  vector<double> Z2(nY);
  vector<int> samples(nY);
  
  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN,
	Kernel::LAPLACIAN }) {
    for (int m : { 1, 13, 101, 765, nX+1, 3*nX-1 }) {
      KdeEstimator::PseudoRandomNumberGenerator rng(nX, seed);
      LinearScan<double> ls(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			   Metric::EUCLIDEAN);
      ls.fit(nX, d, X);

      REQUIRE_THROWS_AS((AnnEstimator<double, LinearScan<double>>(0,kernel,k,m,&ls)),
			std::invalid_argument);
      REQUIRE_THROWS_AS((AnnEstimator<double, LinearScan<double>>(-1,kernel,k,m,&ls)),
			std::invalid_argument);
      REQUIRE_THROWS_AS((AnnEstimator<double, LinearScan<double>>(h,kernel,-1,m,&ls)),
			std::invalid_argument);
      REQUIRE_THROWS_AS((AnnEstimator<double, LinearScan<double>>(h,kernel,k,-1,&ls)),
			std::invalid_argument);
    
      AnnEstimator<double, LinearScan<double>> annEst1(h, kernel, k, m, &ls, seed);
      annEst1.fit(nX, d, X);

      AnnEstimator<double, LinearScan<double>> annEst2(h, kernel, k, m, &ls, seed);
      annEst2.fit(nX, d, X);

      for (int i = 0; i < nY; ++i) {
	double* q = Y + i*d;
	mu1[i] = 0;
	for (int j = 0; j < m; ++j) {
	  uint32_t idx = rng();
	  double* x = X + idx*d;
	  double dist = (kernel == Kernel::LAPLACIAN ?
			array::taxicabDistance(d,q,x,&scratch[0]) :
			array::euclideanDistance(d,q,x,&scratch[0]));
	  mu1[i] += (kernel == Kernel::GAUSSIAN ? std::exp(-dist*dist/h/h/2) :
		     std::exp(-dist/h));
	}
	mu1[i] = mu1[i] * nX / nX / m;
	annEst1.query(d, q, &Z1[i], &samples[i]);
	REQUIRE(Z1[i] == Approx(mu1[i]).epsilon(1e-13));
	REQUIRE(std::abs(Z1[i] - mu1[i]) < 1e-15);
      }
      for (int s : samples)
        REQUIRE(s == m);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      auto rv1 = rng();
      REQUIRE(rv1 == annEst1.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      
      annEst2.query(nY, d, Y, &Z2[0], &samples[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(rv1 == annEst2.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      for (int s : samples)
        REQUIRE(s == m);

      REQUIRE(array::almostEqual(nY, &mu1[0], &Z1[0], 1e-15));
      REQUIRE(array::almostEqual(nY, &mu1[0], &Z2[0], 1e-15));
      REQUIRE(array::equal(nY, &Z1[0], &Z2[0]));
      
      RandomSampling rs(h, kernel, m, seed);
      rs.fit(nX, d, X);
      rs.query(nY, d, Y, &mu2[0], &samples[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(rv1 == rs.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      for (int s : samples)
        REQUIRE(s == m);

      for (int i = 0; i < nY; ++i) {
	REQUIRE(Z1[i] == Approx(mu2[i]).epsilon(1e-13));
	REQUIRE(Z1[i] == Approx(mu2[i]).epsilon(0).margin(1e-16));
	REQUIRE(Z2[i] == Approx(mu2[i]).epsilon(1e-13));
	REQUIRE(Z2[i] == Approx(mu2[i]).epsilon(0).margin(1e-16));
	REQUIRE(std::abs(Z1[i] - mu2[i]) < 1e-15);
      }
      
      REQUIRE(!array::equal(nY, &mu2[0], &Z1[0]));
      REQUIRE(!array::equal(nY, &mu2[0], &Z2[0]));
      REQUIRE(array::almostEqual(nY, &mu2[0], &Z1[0], 1e-15));
      REQUIRE(array::almostEqual(nY, &mu2[0], &Z2[0], 1e-15));
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_ann_estimator5", "[annestimator]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const int seed = 1155487;

  const float h = 30;
  const float DELTA = 1e-6f;
  const float EPSILON = 1e-4f;
  
  vector<float> scratch(d);
  vector<float> mu(nY);
  vector<float> Z(nY);
  vector<int> nns(2*nX,-1);
  vector<float> dists(2*nX,-1);
  vector<int> samples(nY);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN,
	Kernel::LAPLACIAN }) {
    for (int k : { 0, 1, 11, 103, nX, 2*nX }) {
      for (int m : { 0, 1, 7, 100, nX, 2*nX }) {
	KdeEstimator::PseudoRandomNumberGenerator rng(nX, seed);
	LinearScan<float> ls(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			     Metric::EUCLIDEAN);
	ls.fit(nX, d, X);
   
	AnnEstimator<float, LinearScan<float>> annEst(h, kernel, k, m, &ls, seed);
	annEst.fit(nX, d, X);

	for (int i = 0; i < nY; ++i) {
	  float* q = Y + i*d;
	  mu[i] = 0;

	  ls.query(1, d, k, q, &nns[0], &dists[0], &samples[0]);
	  if (k > nX) {
	    for (int i = nX; i < k; ++i) {
	      REQUIRE(nns[i] == -1);
	      REQUIRE(dists[i] == -1);
	    }
	  }
          REQUIRE(samples[0] == (k > 0 ? nX : 0));

	  unordered_set<int> nnsSet(nns.begin(), nns.begin() + std::min(k,nX));

	  if (k == 0)
	    REQUIRE(nnsSet.size() == 0);
	  else
	    REQUIRE(nnsSet.size() == static_cast<size_t>(std::min(k,nX)));
	  REQUIRE(!nnsSet.count(-1));
	  
	  float Z1 = 0;
	  for (auto it = dists.begin(); it != dists.begin() + std::min(k,nX); ++it) {
	    float dist = *it;
	    Z1 += (kernel == Kernel::GAUSSIAN ?
		   std::exp(-dist*dist/h/h/2) : std::exp(-dist/h));
	  }
	  mu[i] = Z1 / nX;
	  
	  float Z2 = 0;
	  if (k < nX && m > 0) {
	    for (int j = 0; j < m; ++j) {
	      uint32_t idx;
	      do {
		idx = rng();
	      }
	      while (nnsSet.count(idx));
	      float* x = X + idx*d;
	      float dist = (kernel == Kernel::LAPLACIAN ?
			    array::taxicabDistance(d,q,x,&scratch[0]) :
			    array::euclideanDistance(d,q,x,&scratch[0]));
	      Z2 += (kernel == Kernel::GAUSSIAN ? std::exp(-dist*dist/h/h/2) :
		     std::exp(-dist/h));
	    }
	    mu[i] += Z2 * (nX-k) / nX / m;
	  }
	  
	  if (k == 0 && m == 0)
	    REQUIRE(mu[i] == 0);
	}
	
	annEst.query(nY, d, Y, &Z[0], &samples[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
	REQUIRE(rng() == annEst.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
        for (int s : samples)
          if (k >= nX)
            REQUIRE(s == nX);
          else if (k > 0)
            REQUIRE(s == nX + m);
          else
            REQUIRE(s == m);

	for (int i = 0; i < nY; ++i) {
	  REQUIRE(mu[i] == Approx(Z[i]).epsilon(EPSILON));
	  REQUIRE(std::abs(mu[i] - Z[i]) < DELTA);
	}
	
	REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));
      }
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_ann_estimator6", "[annestimator]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const int seed = 1256395;

  const double h = 31;
  const double EPSILON = 1e-13;
  const double DELTA = 1e-14;
  
  vector<double> scratch(d);
  vector<double> mu(nY);
  vector<double> Z(nY);
  vector<int> nns(2*nX,-1);
  vector<double> dists(2*nX,-1);
  vector<int> samples(nY);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN,
	Kernel::LAPLACIAN }) {
    for (int k : { 0, 1, 11, 103, nX, 2*nX }) {
      for (int m : { 0, 1, 7, 100, nX, 2*nX }) {
	KdeEstimator::PseudoRandomNumberGenerator rng(nX, seed);
	LinearScan<double> ls(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			     Metric::EUCLIDEAN);
	ls.fit(nX, d, X);
   
	AnnEstimator<double, LinearScan<double>> annEst(h, kernel, k, m, &ls, seed);
	annEst.fit(nX, d, X);

	for (int i = 0; i < nY; ++i) {
	  double* q = Y + i*d;
	  mu[i] = 0;

	  ls.query(1, d, k, q, &nns[0], &dists[0], &samples[0]);
	  if (k > nX) {
	    for (int i = nX; i < k; ++i) {
	      REQUIRE(nns[i] == -1);
	      REQUIRE(dists[i] == -1);
	    }
	  }
          REQUIRE(samples[0] == (k > 0 ? nX : 0));
	  unordered_set<int> nnsSet(nns.begin(), nns.begin() + std::min(k,nX));

	  if (k == 0)
	    REQUIRE(nnsSet.size() == 0);
	  else
	    REQUIRE(nnsSet.size() == static_cast<size_t>(std::min(k,nX)));
	  
	  double Z1 = 0;
	  for (auto it = dists.begin(); it != dists.begin() + std::min(k,nX); ++it) {
	    double dist = *it;
	    Z1 += (kernel == Kernel::GAUSSIAN ?
		   std::exp(-dist*dist/h/h/2) : std::exp(-dist/h));
	  }
	  mu[i] = Z1 / nX;
	  
	  double Z2 = 0;
	  if (k < nX && m > 0) {
	    for (int j = 0; j < m; ++j) {
	      uint32_t idx;
	      do {
		idx = rng();
	      }
	      while (nnsSet.count(idx));
	      double* x = X + idx*d;
	      double dist = (kernel == Kernel::LAPLACIAN ?
			    array::taxicabDistance(d,q,x,&scratch[0]) :
			    array::euclideanDistance(d,q,x,&scratch[0]));
	      Z2 += (kernel == Kernel::GAUSSIAN ? std::exp(-dist*dist/h/h/2) :
		     std::exp(-dist/h));
	    }
	    mu[i] += Z2 * (nX-k) / nX / m;
	  }
	  
	  if (k == 0 && m == 0)
	    REQUIRE(mu[i] == 0);
	}
	
	annEst.query(nY, d, Y, &Z[0], &samples[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
	REQUIRE(rng() == annEst.getRng()());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
        for (int s : samples)
          if (k >= nX)
            REQUIRE(s == nX);
          else if (k > 0)
            REQUIRE(s == nX + m);
          else
            REQUIRE(s == m);

	for (int i = 0; i < nY; ++i) {
	  REQUIRE(mu[i] == Approx(Z[i]).epsilon(EPSILON));
	  REQUIRE(std::abs(mu[i] - Z[i]) < DELTA);
	}
	
	REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));
      }
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_ann_estimator7", "[annestimator]" ) {
  const double h = 16;
  float* data1;
  double* data2;
  int n, d;
  constructMediumTestSet(&data1, &n, &d);

  float *X1, *Y1;
  double *X2, *Y2;
  int nX, nY;
  constructMediumTestSet(&data2, &n, &d);
  array::splitInHalf(n,d,&nX,&nY,data1,&X1,&Y1);
  array::splitInHalf(n,d,&nX,&nY,data2,&X2,&Y2);

  LinearScan<float> ls1(Metric::EUCLIDEAN);
  ls1.fit(nX, d, X1);
  LinearScan<double> ls2(Metric::EUCLIDEAN);
  ls2.fit(nX, d, X2);

  AnnEstimator<float, LinearScan<float>> ann1(h, Kernel::EXPONENTIAL, 0, 0, &ls1);
  AnnEstimator<double, LinearScan<double>> ann2(h, Kernel::EXPONENTIAL, 0, 0, &ls2);

  REQUIRE_NOTHROW(ann1.resetParameters());
  REQUIRE_NOTHROW(ann1.resetParameters(0));
  REQUIRE_NOTHROW(ann1.resetParameters(1));
  REQUIRE_NOTHROW(ann1.resetParameters(0,nullopt));
  REQUIRE_NOTHROW(ann1.resetParameters(1,nullopt));
  REQUIRE_NOTHROW(ann1.resetParameters(0,0));
  REQUIRE_NOTHROW(ann1.resetParameters(1,1));
  REQUIRE_NOTHROW(ann1.resetParameters(nullopt,0));
  REQUIRE_NOTHROW(ann1.resetParameters(nullopt,1));
  REQUIRE_NOTHROW(ann1.resetParameters(nullopt,nullopt));
  
  REQUIRE_THROWS_AS(ann1.resetParameters(-1), invalid_argument);
  REQUIRE_THROWS_AS(ann1.resetParameters(-1,nullopt), invalid_argument);
  REQUIRE_THROWS_AS(ann1.resetParameters(nullopt,-1), invalid_argument);
  REQUIRE_THROWS_AS(ann1.resetParameters(-1,-1), invalid_argument);

  REQUIRE_NOTHROW(ann2.resetParameters());
  REQUIRE_NOTHROW(ann2.resetParameters(0));
  REQUIRE_NOTHROW(ann2.resetParameters(1));
  REQUIRE_NOTHROW(ann2.resetParameters(0,nullopt));
  REQUIRE_NOTHROW(ann2.resetParameters(1,nullopt));
  REQUIRE_NOTHROW(ann2.resetParameters(0,0));
  REQUIRE_NOTHROW(ann2.resetParameters(1,1));
  REQUIRE_NOTHROW(ann2.resetParameters(nullopt,0));
  REQUIRE_NOTHROW(ann2.resetParameters(nullopt,1));
  REQUIRE_NOTHROW(ann2.resetParameters(nullopt,nullopt));
  
  REQUIRE_THROWS_AS(ann2.resetParameters(-1), invalid_argument);
  REQUIRE_THROWS_AS(ann2.resetParameters(-1,nullopt), invalid_argument);
  REQUIRE_THROWS_AS(ann2.resetParameters(nullopt,-1), invalid_argument);
  REQUIRE_THROWS_AS(ann2.resetParameters(-1,-1), invalid_argument);

  ann1 = AnnEstimator<float,LinearScan<float>>(h, Kernel::EXPONENTIAL, 0, 0, &ls1);
  ann2 = AnnEstimator<double,LinearScan<double>>(h, Kernel::EXPONENTIAL, 0, 0, &ls2);
  ann1.fit(nX, d, X1);
  ann2.fit(nX, d, X2);
  vector<float> Z1_1(nY);
  vector<double> Z2_1(nY);
  vector<float> Z1_2(nY);
  vector<double> Z2_2(nY);
  vector<int> samples(nY);

  const int seed = 8520;

  for (int k : { 0, 11, 22, 33 }) {
    for (int m : { 0, 123, 234, 345 }) {
      ann1.resetParameters(k, m);
      ann1.resetSeed(seed);
      ann1.query(nY, d, Y1, &Z1_1[0], &samples[0]);
      for (int s : samples)
        REQUIRE(s == (k > 0 ? nX : 0) + m);

      ann2.resetParameters(k, m);
      ann2.resetSeed(seed);
      ann2.query(nY, d, Y2, &Z2_1[0], &samples[0]);
      for (int s : samples)
        REQUIRE(s == (k > 0 ? nX : 0) + m);
      
      AnnEstimator<float,LinearScan<float>> ann3(h, Kernel::EXPONENTIAL, k, m, &ls1, seed);
      ann3.fit(nX, d, X1);
      ann3.query(nY, d, Y1, &Z1_2[0], &samples[0]);
      for (int s : samples)
        REQUIRE(s == (k > 0 ? nX : 0) + m);

      REQUIRE(array::equal(nY, &Z1_1[0], &Z1_2[0]));

      AnnEstimator<double,LinearScan<double>> ann4(h, Kernel::EXPONENTIAL, k, m, &ls2, seed);
      ann4.fit(nX, d, X2);
      ann4.query(nY, d, Y2, &Z2_2[0], &samples[0]);    
      for (int s : samples)
        REQUIRE(s == (k > 0 ? nX : 0) + m);
      REQUIRE(array::equal(nY, &Z2_1[0], &Z2_2[0]));
    }
  }
  
  allocator.free(data1);
  allocator.free(data2);
}



TEST_CASE( "test_ann_estimator8", "[annestimator]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const int seed = 0xeef5d8dc;
  const float h = 25;
  const float EPSILON = 1e-4;
  const float DELTA = 1e-6;

  vector<float> Z1(nY);
  vector<float> Z2(nY);
  vector<int> samples(nY);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN,
	Kernel::LAPLACIAN }) {
    for (int k : { 0, 1, 33, nX }) {
      for (int m : { 0, 1, 47, nX }) {
	LinearScan<float> ls1(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			      Metric::EUCLIDEAN, true);
	ls1.fit(nX, d, X);
	LinearScan<float> ls2(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			      Metric::EUCLIDEAN, false);
	ls2.fit(nX, d, X);
	AnnEstimator<float, LinearScan<float>> annEst1(h, kernel, k, m, &ls1, seed);
	annEst1.fit(nX, d, X);
	AnnEstimator<float, LinearScan<float>> annEst2(h, kernel, k, m, &ls2, seed);
	annEst2.fit(nX, d, X);

	annEst1.query(nY, d, Y, &Z1[0], &samples[0]);
        for (int s : samples)
          if (k >= nX)
            REQUIRE(s == nX);
          else if (k > 0)
            REQUIRE(s == nX + m);
          else
            REQUIRE(s == m);
	annEst2.query(nY, d, Y, &Z2[0], &samples[0]);
        for (int s : samples)
          if (k >= nX)
            REQUIRE(s == nX);
          else if (k > 0)
            REQUIRE(s == nX + m);
          else
            REQUIRE(s == m);
        
	if (k == 0 && m == 0) {
	  REQUIRE(array::allEqual(nY, &Z1[0], 0.0f));
	  REQUIRE(array::allEqual(nY, &Z2[0], 0.0f));
	}
	else {
	  for (int i = 0; i < nY; ++i) {
	    REQUIRE(Z1[i] == Approx(Z2[i]).epsilon(EPSILON));
	    REQUIRE(Z1[i] == Approx(Z2[i]).epsilon(0).margin(DELTA));
	    REQUIRE(Z1[i] > 0);
	    REQUIRE(Z2[i] > 0);
	  }
	  REQUIRE(array::almostEqual(nY, &Z1[0], &Z2[0], DELTA));
	}
      }
    }
  }
  allocator.free(data);
}


  
TEST_CASE( "test_ann_estimator9", "[annestimator]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  const int seed = 0x96a3d6ab;
  const double h = 22;
  const double EPSILON = 1e-13;
  const double DELTA = 1e-15;

  vector<double> Z1(nY);
  vector<double> Z2(nY);
  vector<int> samples(nY);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN,
	Kernel::LAPLACIAN }) {
    for (int k : { 0, 1, 33, nX }) {
      for (int m : { 0, 1, 47, nX }) {
	LinearScan<double> ls1(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			      Metric::EUCLIDEAN, true);
	ls1.fit(nX, d, X);
	LinearScan<double> ls2(kernel == Kernel::LAPLACIAN ? Metric::TAXICAB :
			      Metric::EUCLIDEAN, false);
	ls2.fit(nX, d, X);
	AnnEstimator<double, LinearScan<double>> annEst1(h, kernel, k, m, &ls1, seed);
	annEst1.fit(nX, d, X);
	AnnEstimator<double, LinearScan<double>> annEst2(h, kernel, k, m, &ls2, seed);
	annEst2.fit(nX, d, X);

	annEst1.query(nY, d, Y, &Z1[0], &samples[0]);
        if (k >= nX)
          REQUIRE(array::allEqual(nY, &samples[0], nX));
        else if (k > 0)
          REQUIRE(array::allEqual(nY, &samples[0], nX + m));
        else
          REQUIRE(array::allEqual(nY, &samples[0], m));
	annEst2.query(nY, d, Y, &Z2[0], &samples[0]);
        if (k >= nX)
          REQUIRE(array::allEqual(nY, &samples[0], nX));
        else if (k > 0)
          REQUIRE(array::allEqual(nY, &samples[0], nX + m));
        else
          REQUIRE(array::allEqual(nY, &samples[0], m));
        
	if (k == 0 && m == 0) {
	  REQUIRE(array::allEqual(nY, &Z1[0], 0.0));
	  REQUIRE(array::allEqual(nY, &Z2[0], 0.0));
	}
	else {
	  for (int i = 0; i < nY; ++i) {
	    REQUIRE(Z1[i] == Approx(Z2[i]).epsilon(EPSILON));
	    REQUIRE(Z1[i] == Approx(Z2[i]).epsilon(0).margin(DELTA));
	    REQUIRE(Z1[i] > 0);
	    REQUIRE(Z2[i] > 0);
	  }
	  REQUIRE(array::almostEqual(nY, &Z1[0], &Z2[0], DELTA));
	}
      }
    }
  }
  allocator.free(data);
}



TEST_CASE( "test_hash1", "[rng]" ) {
  uint64_t a = 0xff32352db88874c9;
  uint64_t b = 0x169d63bf1db62c13;

  uint32_t m = 0x100;
  REQUIRE(hash_to_range(0, m, a, b) != 0);

  int spots256[256];
  for (int i = 0; i < 256; ++i)
    spots256[i] = 0;
  
  for (uint32_t i = 0; i < 10*m; ++i) {
    uint32_t h = hash_to_range(i, m, a, b);
    REQUIRE(h < m);
    ++spots256[h];
  }

  for (int i = 0; i < 256; ++i) {
    REQUIRE(spots256[i] >= 8);
    REQUIRE(spots256[i] <= 12);
  }

  uint32_t x = hash_to_range(m, ~0, a, b);
  for (uint32_t m : { 100, 234, 555, 1234 }) {
    vector<int> spots(m,0);
    for (uint32_t i = 0; i < 100*m; ++i) {
      uint32_t h = hash_to_range(x, m, a, b);
      REQUIRE(h < m);
      ++spots[h];
      ++x;
    }
    for (uint32_t i = 0; i < m; ++i) {
      REQUIRE(spots[i] >= 97);
      REQUIRE(spots[i] <= 103);
    }
  }
}



TEST_CASE( "test_mt19937_rng", "[rng]" ) {
  static_assert(sizeof(uint64_t) == sizeof(1ull));

#if 0
  // this takes quite a bit of time, so the test is disabled for now
  vector<char> found(1ull << 32, 0);
  REQUIRE(found.size() == 4294967296);
  uint64_t* ptr = reinterpret_cast<uint64_t*>(&found[0]);
  for (uint64_t i = 0; i < found.size() / sizeof(uint64_t); ++i)
    REQUIRE(ptr[i] == 0ull);

  Mt19937Rng rng(0, 1234);

  for (uint64_t i = 0; i < 23*found.size(); ++i) {
    found[rng()] = 1;
  }

  uint64_t target = 1ull | (1ull << 8);
  target |= (target << 16);
  target |= (target << 32);

  for (uint64_t i = 0; i < found.size()/sizeof(uint64_t); ++i) {
    if (ptr[i] != target) {
      cerr << i << " " << i*sizeof(uint64_t) << endl;
      fprintf(stderr, "%016llx\n", ptr[i]);
    }
    REQUIRE(ptr[i] == target);
  }
#endif

  
  for (int m : { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 51, 101, 503, 1000,
	1000000 }) {
    uint32_t seed = mt19937(m)();
    vector<int> found(m,0);
    Mt19937Rng rng(m, seed);
    for (int i = 0; i < 17*m; ++i) {
      uint32_t idx = rng();
      REQUIRE(idx < static_cast<uint32_t>(m));
      ++found[idx];
    }
    for (int i = 0; i < m; ++i) {
      if (found[i] == 0)
	cerr << i << "/" << m << " " << found[i] << endl;
      REQUIRE(found[i] > 0);
    }
  }
}



TEST_CASE( "test_fast_rng", "[rng]" ) {
  REQUIRE(gcd(1071,462) == 21);
  for (uint64_t x = 1; x < 103; ++x) {
    REQUIRE(gcd(x,103) == 1);
    REQUIRE(gcd(x,x) == x);
  }
  for (uint64_t x : { 2, 7, 17 }) {
    for (uint64_t y : { 3, 11, 19 }) {
      for (uint64_t z : { 5, 13, 23 }) {
	REQUIRE(gcd(x*y,y*z) == y);
      }
    }
  }

  const uint32_t seedseed = 0x277c365d;
  mt19937 mtrng(seedseed);

  uint32_t seed = mtrng();
  REQUIRE_THROWS_AS(FastRng(0, seed), invalid_argument);

  for (uint32_t m : { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 21, 35, 74, 100, 1100,
	12345, 100000 }) {
    seed = mtrng();
    FastRng rng(m, seed);
    FastRng rng2(m, seed);
    mt19937 mtrng2(seed);

    uint64_t a; 
    do {
      a = mtrng2();
      a = (a << 32) | mtrng2();
    }
    while (gcd(a,m) != 1);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(a == rng.a);
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(gcd(a,m) == 1);

    uint64_t b; 
    do {
      b = mtrng2();
      b = (b << 32) | mtrng2();
    }
    while (gcd(b,m) != 1);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(b == rng.b);
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(gcd(b,m) == 1);

    uint32_t x = mtrng2();
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(x == rng.x);
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

    uint64_t niters = 2000ull*m;
    vector<uint32_t> randints(niters);
    vector<uint32_t> randints2(niters);

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
    const int TIME_REPEATS = 5;
    
    for (uint64_t i = 0; i < niters; ++i)
      randints[i] = 0;

    for (int rep = 0; rep < TIME_REPEATS; ++rep) {
      timerOn();
      mt19937 mtrng3;
      uniform_int_distribution<uint32_t> ud(0,m-1);
      for (uint64_t i = 0; i < niters; ++i) {
	randints[i] = ud(mtrng3);
      }
      auto diff = timerOff();
      cerr << "mt19937 m=" << m << ", niters=" << niters << " took "
	   << parseDuration(diff) << endl;
    }
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
    for (int rep = 0; rep < TIME_REPEATS; ++rep) {
      timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

      for (uint64_t i = 0; i < niters; ++i) {
	uint32_t h = rng();
	randints[i] = h;
      }
    
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
      auto diff = timerOff();
      cerr << "fastRng m=" << m << ", niters=" << niters << " took "
	   << parseDuration(diff) << endl;
    }
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
    for (uint64_t i = 0; i < niters; ++i)
      randints2[i] = 0;

    for (int rep = 0; rep < TIME_REPEATS; ++rep) {
      timerOn();
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES

      rng2(niters, &randints2[0]);
    
#ifdef DEANN_TESTS_DEBUG_PRINT_TIMES
      auto diff = timerOff();
      cerr << "fastRng batch m=" << m << ", niters=" << niters << " took "
	   << parseDuration(diff) << endl;
    }
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
    
#ifndef DEANN_TESTS_DEBUG_PRINT_TIMES
    vector<uint64_t> counts(m,0);
    for (uint64_t i = 0; i < niters; ++i) {
      uint32_t h = randints[i];
      REQUIRE(h < m);
      REQUIRE(hash_to_range(x++, m, a, b) == h);
      REQUIRE(randints2[i] == h);
      ++counts[h];

    }

    if (m == 1)
      REQUIRE(counts[0] == niters);
    else {
      for (uint32_t i = 0; i < m; ++i) {
	REQUIRE(counts[i] > 0);
	double r = static_cast<double>(counts[i]) / niters;
	double relerr = std::abs(r-1.0/m)*m;
	if (relerr > 0.01)
	  cerr << i << " " << r << " " << relerr << endl;
	REQUIRE(relerr < 0.01);
      }
    }
#endif // DEANN_TESTS_DEBUG_PRINT_TIMES
  }
}


TEST_CASE( "test_random_sampling_permuted1", "[randomsamplingpermuted]" ) {
  const uint32_t seedseed = 0xe6408d13;
  mt19937 seedRng(seedseed);

  const float h = 16;
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);

  const int m = 100;

  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(0.0, Kernel::EXPONENTIAL, m),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(-1, Kernel::EXPONENTIAL, m),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(h, Kernel::LAPLACIAN, m),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(h, Kernel::EXPONENTIAL, 0),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(h, Kernel::EXPONENTIAL, -1),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(h, Kernel::EXPONENTIAL, 123)
		    .fit(122, 11, X), invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(h, Kernel::EXPONENTIAL, 123)
		    .fit(0, 11, X), invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(h, Kernel::EXPONENTIAL, 123)
		    .fit(-1, 11, X), invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(h, Kernel::EXPONENTIAL, 123)
		    .fit(123, 0, X), invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<float>(h, Kernel::EXPONENTIAL, 123)
		    .fit(123, -1, X), invalid_argument);

  const float DELTA = 1e-6f;
  
  vector<float> scratch(d);
  
  vector<float> mu(nY);
  vector<float> Z1(nY);
  vector<float> Z2(nY);
  vector<float> Z3(nY);

  vector<int> samples(nY);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN }) {
    uint32_t seed = seedRng();
    RandomSamplingPermuted<float> rsp1(h, kernel, 1, seed);
    rsp1.fit(nX, d, X);

    mt19937 rng(seed);
    vector<uint32_t> idx(nX);
    for (uint32_t i = 0; i < static_cast<uint32_t>(nX); ++i)
      idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    const float* Xp = rsp1.getXpermuted();
    for (int i = 0; i < nX; ++i) {
      float* x = X + idx[i]*d;
      REQUIRE(array::equal(d, x, Xp + i*d));
    }
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

    rsp1.resetParameters(nX);
    rsp1.query(nY, d, Y, &Z1[0], &samples[0]);
    
    for (int s : samples)
      REQUIRE(s == nX);

    RandomSamplingPermuted<float> rsp2(h, kernel, nX, seed);
    rsp2.fit(nX, d, X);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    const float* Xp2 = rsp2.getXpermuted();
    for (int i = 0; i < nX*d; ++i)
      REQUIRE(Xp[i] == Xp2[i]);

    for (int i = 0; i < nX; ++i)
      REQUIRE(rsp1.getXSqNorm()[i] == rsp2.getXSqNorm()[i]);
    
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    
    rsp2.query(nY, d, Y, &Z2[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nX);


    for (int i = 0; i < nY; ++i) {
      // by all reason these should compare equal, but under some
      // circumstances this does not hold
      // this is probably some weird MKL feature
      // REQUIRE(Z1[i] == Z2[i]);
      REQUIRE(Z1[i] == Approx(Z2[i]).epsilon(1e-4f));
    }
      
    // REQUIRE(array::equal(nY, &Z1[0], &Z2[0]));
    REQUIRE(array::almostEqual(nY, &Z1[0], &Z2[0], 1e-4f));

    RandomSamplingPermuted<float> rsp3(h, kernel, nX, seed);
    rsp3.fit(nX, d, X);
    for (int i = 0; i < nY; ++i) {
      float* y = Y + i*d;
      rsp3.query(d, y, &Z3[i], &samples[i]);
    }
    for (int s : samples)
      REQUIRE(s == nX);
    // REQUIRE(array::equal(nY, &Z1[0], &Z3[0]));
    
    // these should compare equal as well but again something
    // suspicious is going on with MKL
    REQUIRE(array::almostEqual(nY, &Z1[0], &Z3[0], 1e-4f));
    REQUIRE(array::almostEqual(nY, &Z2[0], &Z3[0], 1e-4f));

    NaiveKde<float> nkde(h, kernel);
    nkde.fit(nX, d, X);
    nkde.query(nY, d, Y, &mu[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nX);

    for (int i = 0; i < nY; ++i) {
      REQUIRE(std::abs(mu[i] - Z1[i]) < DELTA);
    }
    REQUIRE(array::almostEqual(nY, &mu[0], &Z1[0], DELTA));

    int k = 0;
    for (int i = 0; i < nY; ++i) {
      float* q = Y + i*d;
      mu[i] = 0;
      for (int j = 0; j < m; ++j) {
	float* x = X + idx[k]*d;
	float dist = array::euclideanDistance(d, x, q, &scratch[0]);
	if (kernel == Kernel::EXPONENTIAL)
	  mu[i] += std::exp(-dist/h);
	if (kernel == Kernel::GAUSSIAN)
	  mu[i] += std::exp(-dist*dist/h/h/2);
	k = (k+1) % nX;
      }
      mu[i] /= m;
    }
    
    rsp1 = RandomSamplingPermuted<float>(h, kernel, m, seed);
    rsp1.fit(nX, d, X);
    rsp1.query(nY, d, Y, &Z1[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == m);

    rsp2 = RandomSamplingPermuted<float>(h, kernel, m, seed);
    rsp2.fit(nX, d, X);
    for (int i = 0; i < nY; ++i)
      rsp2.query(d, Y + i*d, &Z2[i], &samples[i]);
    for (int s : samples)
      REQUIRE(s == m);

    for (int i = 0; i < nY; ++i) {
      // REQUIRE(Z1[i] == Z2[i]);
      REQUIRE(Z1[i] == Approx(Z2[i]).epsilon(1e-4f));
      REQUIRE(std::abs(mu[i] - Z1[i]) < DELTA);
    }
    REQUIRE(array::almostEqual(nY, &mu[0], &Z1[0], DELTA));
  }

  allocator.free(data);
}



TEST_CASE( "test_random_sampling_permuted2", "[randomsamplingpermuted]" ) {
  const uint32_t seedseed = 0xb9bc8082;
  mt19937 seedRng(seedseed);

  const double h = 27;
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);

  const int m = 103;

  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(0.0, Kernel::EXPONENTIAL, m),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(-1, Kernel::EXPONENTIAL, m),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(h, Kernel::LAPLACIAN, m),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(h, Kernel::EXPONENTIAL, 0),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(h, Kernel::EXPONENTIAL, -1),
		    invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(h, Kernel::EXPONENTIAL, 123)
		    .fit(122, 11, X), invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(h, Kernel::EXPONENTIAL, 123)
		    .fit(0, 11, X), invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(h, Kernel::EXPONENTIAL, 123)
		    .fit(-1, 11, X), invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(h, Kernel::EXPONENTIAL, 123)
		    .fit(123, 0, X), invalid_argument);
  REQUIRE_THROWS_AS(RandomSamplingPermuted<double>(h, Kernel::EXPONENTIAL, 123)
		    .fit(123, -1, X), invalid_argument);

  const double DELTA = 1e-14;
  
  vector<double> scratch(d);
  
  vector<double> mu(nY);
  vector<double> Z1(nY);
  vector<double> Z2(nY);
  vector<double> Z3(nY);
  vector<int> samples(nY);

  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN }) {
    uint32_t seed = seedRng();
    RandomSamplingPermuted<double> rsp1(h, kernel, 1, seed);
    rsp1.fit(nX, d, X);

    mt19937 rng(seed);
    vector<uint32_t> idx(nX);
    for (uint32_t i = 0; i < static_cast<uint32_t>(nX); ++i)
      idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    const double* Xp = rsp1.getXpermuted();
    for (int i = 0; i < nX; ++i) {
      double* x = X + idx[i]*d;
      REQUIRE(array::equal(d, x, Xp + i*d));
    }
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

    rsp1.resetParameters(nX);
    rsp1.query(nY, d, Y, &Z1[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nX);

    RandomSamplingPermuted<double> rsp2(h, kernel, nX, seed);
    rsp2.fit(nX, d, X);

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    const double* Xp2 = rsp2.getXpermuted();
    for (int i = 0; i < nX*d; ++i)
      REQUIRE(Xp[i] == Xp2[i]);
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    
    rsp2.query(nY, d, Y, &Z2[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nX);


    for (int i = 0; i < nY; ++i) {
      REQUIRE(Z1[i] == Z2[i]);
    }
      

    REQUIRE(array::equal(nY, &Z1[0], &Z2[0]));

    RandomSamplingPermuted<double> rsp3(h, kernel, nX, seed);
    rsp3.fit(nX, d, X);
    for (int i = 0; i < nY; ++i) {
      double* y = Y + i*d;
      rsp3.query(d, y, &Z3[i], &samples[i]);
    }
    for (int s : samples)
      REQUIRE(s == nX);
    REQUIRE(array::equal(nY, &Z1[0], &Z3[0]));

    NaiveKde<double> nkde(h, kernel);
    nkde.fit(nX, d, X);
    nkde.query(nY, d, Y, &mu[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == nX);


    for (int i = 0; i < nY; ++i) {
      REQUIRE(std::abs(mu[i] - Z1[i]) < DELTA);
    }
    REQUIRE(array::almostEqual(nY, &mu[0], &Z1[0], DELTA));

    int k = 0;
    for (int i = 0; i < nY; ++i) {
      double* q = Y + i*d;
      mu[i] = 0;
      for (int j = 0; j < m; ++j) {
	double* x = X + idx[k]*d;
	double dist = array::euclideanDistance(d, x, q, &scratch[0]);
	if (kernel == Kernel::EXPONENTIAL)
	  mu[i] += std::exp(-dist/h);
	if (kernel == Kernel::GAUSSIAN)
	  mu[i] += std::exp(-dist*dist/h/h/2);
	k = (k+1) % nX;
      }
      mu[i] /= m;
    }
    
    rsp1 = RandomSamplingPermuted<double>(h, kernel, m, seed);
    rsp1.fit(nX, d, X);
    rsp1.query(nY, d, Y, &Z1[0], &samples[0]);
    for (int s : samples)
      REQUIRE(s == m);


    rsp2 = RandomSamplingPermuted<double>(h, kernel, m, seed);
    rsp2.fit(nX, d, X);
    for (int i = 0; i < nY; ++i)
      rsp2.query(d, Y + i*d, &Z2[i], &samples[i]);
    for (int s : samples)
      REQUIRE(s == m);

    for (int i = 0; i < nY; ++i) {
      REQUIRE(Z1[i] == Z2[i]);
      REQUIRE(std::abs(mu[i] - Z1[i]) < DELTA);
    }
    REQUIRE(array::almostEqual(nY, &mu[0], &Z1[0], DELTA));
  }

  allocator.free(data);
}



TEST_CASE( "test_random_sampling_permuted3", "[randomsamplingpermuted]" ) {
  const uint32_t seed = 0xdfd3f30d;

  const float h = 27;
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);

  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n,d,&nX,&nY,data,&X,&Y);

  RandomSamplingPermuted rsp(h, Kernel::EXPONENTIAL, 2*nX, seed);
  REQUIRE_THROWS_AS(rsp.fit(nX, d, X), invalid_argument);
  REQUIRE_THROWS_AS(rsp.setRandomSamples(2*nX), invalid_argument);
  REQUIRE_THROWS_AS(rsp.resetParameters(2*nX), invalid_argument);
  allocator.free(data);
}



TEST_CASE( "test_ann_estimator_permuted3", "[annestimatorpermuted]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  LinearScan<float> ls(Metric::EUCLIDEAN);
  ls.fit(nX, d, X);

  float h = 25;

  KdeEstimator::PseudoRandomNumberGenerator::ValueType seed = 0xb2dd92eb;

  AnnEstimatorPermuted<float,LinearScan<float>> aep(h, Kernel::EXPONENTIAL, nX,
                                                    nX, &ls, seed);
  REQUIRE_THROWS_AS(aep.fit(nX, d, X), invalid_argument);
  aep.resetParameters(nX/2, nX/2);
  aep.fit(nX, d, X);
  REQUIRE_THROWS_AS(aep.setRandomSamples(nX), invalid_argument);
  REQUIRE_THROWS_AS(aep.setNearNeighbors(nX), invalid_argument);
  REQUIRE_THROWS_AS(aep.resetParameters(nX), invalid_argument);
  REQUIRE_THROWS_AS(aep.resetParameters(std::nullopt, nX), invalid_argument);
  aep.setRandomSamples(nX-nX/2);
  aep.setRandomSamples(nX/2);
  aep.setNearNeighbors(nX-nX/2);
  aep.setNearNeighbors(nX/2);
  aep.resetParameters(nX-2,2);

  allocator.free(data);
}



TEST_CASE( "test_missing", "[missing]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
    float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);
  
  int seed = 1234;
  float h = 34;

  vector<float> mu1(nY);
  vector<float> mu2(nY);
  vector<float> Z1(nY);
  vector<float> Z2(nY);
  vector<float> Z3(nY);
  vector<float> Z4(nY);
  vector<int> samples1(nY);
  vector<int> samples2(nY);
  vector<int> samples3(nY);
  vector<int> samples4(nY);
    
  
  NaiveKde<float> nkde(h, Kernel::EXPONENTIAL);
  nkde.fit(nX, d, X);
  nkde.query(nY, d, Y, &Z1[0], &samples1[0]);
  nkde.query(nY, d, Y, &Z2[0], nullptr);
  REQUIRE(array::allEqual(nY, &samples1[0], nX));
  REQUIRE(array::equal(nY, &Z1[0], &Z2[0]));

  float EPSILON = 1e-4f;
  
  float Z;
  int S;
  for (int i = 0; i < nY; ++i) {
    nkde.query(d, Y + i*d, &Z, &S);
    REQUIRE(S == nX);
    REQUIRE(Z == Approx(Z1[i]).epsilon(EPSILON));
    nkde.query(d, Y + i*d, &Z, nullptr);
    REQUIRE(Z == Approx(Z1[i]).epsilon(EPSILON));
  }

  int m = 123;
  
  RandomSampling<float> rs1(h, Kernel::EXPONENTIAL, m, seed);
  RandomSampling<float> rs2(h, Kernel::EXPONENTIAL, m, seed);
  rs1.fit(nX, d, X);
  rs2.fit(nX, d, X);
  rs1.query(nY, d, Y, &Z1[0], &samples1[0]);
  rs2.query(nY, d, Y, &Z2[0], nullptr);
  REQUIRE(array::allEqual(nY, &samples1[0], m));
  REQUIRE(array::equal(nY, &Z1[0], &Z2[0]));
  rs1.resetSeed(seed);
  rs2.resetSeed(seed);
  for (int i = 0; i < nY; ++i) {
    rs1.query(d, Y + i*d, &Z, &S);
    REQUIRE(S == m);
    REQUIRE(Z == Z1[i]);
    rs2.query(d, Y + i*d, &Z, nullptr);
    REQUIRE(Z == Z1[i]);
  }

  RandomSampling<float> rsp1(h, Kernel::EXPONENTIAL, m, seed);
  RandomSampling<float> rsp2(h, Kernel::EXPONENTIAL, m, seed);
  rsp1.fit(nX, d, X);
  rsp2.fit(nX, d, X);
  rsp1.query(nY, d, Y, &Z1[0], &samples1[0]);
  rsp2.query(nY, d, Y, &Z2[0], nullptr);
  REQUIRE(array::allEqual(nY, &samples1[0], m));
  REQUIRE(array::equal(nY, &Z1[0], &Z2[0]));
  rsp1.resetSeed(seed);
  rsp2.resetSeed(seed);
  for (int i = 0; i < nY; ++i) {
    rsp1.query(d, Y + i*d, &Z, &S);
    REQUIRE(S == m);
    REQUIRE(Z == Z1[i]);
    rsp2.query(d, Y + i*d, &Z, nullptr);
    REQUIRE(Z == Z1[i]);
  }

  for (int k : { 0, 123, nX }) {
    for (int m : { 0, 123 }) {
      LinearScan<float> ls1(Metric::EUCLIDEAN, true, true);
      LinearScan<float> ls2(Metric::EUCLIDEAN, false, true);
      LinearScan<float> ls3(Metric::EUCLIDEAN, true, false);
      LinearScan<float> ls4(Metric::EUCLIDEAN, false, false);
      
      ls1.fit(nX, d, X);
      ls2.fit(nX, d, X);
      ls3.fit(nX, d, X);
      ls4.fit(nX, d, X);

      AnnEstimator<float,LinearScan<float>> ann1(h, Kernel::EXPONENTIAL, k, m, &ls1, seed);
      AnnEstimator<float,LinearScan<float>> ann2(h, Kernel::EXPONENTIAL, k, m, &ls2, seed);
      AnnEstimator<float,LinearScan<float>> ann3(h, Kernel::EXPONENTIAL, k, m, &ls3, seed);
      AnnEstimator<float,LinearScan<float>> ann4(h, Kernel::EXPONENTIAL, k, m, &ls4, seed);

      ann1.fit(nX, d, X);
      ann2.fit(nX, d, X);
      ann3.fit(nX, d, X);
      ann4.fit(nX, d, X);

      ann1.query(nY, d, Y, &Z1[0], &samples1[0]);
      ann2.query(nY, d, Y, &Z2[0], &samples2[0]);
      ann3.query(nY, d, Y, &Z3[0], &samples3[0]);
      ann4.query(nY, d, Y, &Z4[0], &samples4[0]);
      
      array::mov(nY, &Z1[0], &mu1[0]);
      
      if (k == 0) {
        array::mov(nY, &Z1[0], &mu2[0]);
        REQUIRE(array::equal(nY, &mu1[0], &mu2[0]));
      }
      else {
        array::mov(nY, &Z2[0], &mu2[0]);
        REQUIRE(array::almostEqual(nY, &mu1[0], &mu2[0], EPSILON));
      }

      if (k == 0 && m == 0)
        REQUIRE(array::allEqual(nY, &mu1[0], 0.0f));

      REQUIRE(array::equal(nY, &mu1[0], &Z1[0]));
      REQUIRE(array::equal(nY, &mu2[0], &Z2[0]));
      REQUIRE(array::equal(nY, &mu1[0], &Z3[0]));
      REQUIRE(array::equal(nY, &mu2[0], &Z4[0]));

      REQUIRE(array::allEqual(nY, &samples1[0],
                              k >= nX ? nX :
                              k > 0 ? nX + m :
                              m));
      REQUIRE(array::allEqual(nY, &samples2[0],
                              k >= nX ? nX :
                              k > 0 ? nX + m :
                              m));
      REQUIRE(array::allEqual(nY, &samples3[0], k > 0 ? -1 : m));
      REQUIRE(array::allEqual(nY, &samples4[0], k > 0 ? -1 : m));

      ann1.resetSeed(seed);
      ann2.resetSeed(seed);
      ann3.resetSeed(seed);
      ann4.resetSeed(seed);

      ann1.query(nY, d, Y, &Z1[0], nullptr);
      ann2.query(nY, d, Y, &Z2[0], nullptr);
      ann3.query(nY, d, Y, &Z3[0], nullptr);
      ann4.query(nY, d, Y, &Z4[0], nullptr);

      REQUIRE(array::equal(nY, &mu1[0], &Z1[0]));
      REQUIRE(array::equal(nY, &mu2[0], &Z2[0]));
      REQUIRE(array::equal(nY, &mu1[0], &Z3[0]));
      REQUIRE(array::equal(nY, &mu2[0], &Z4[0]));

      ann1.resetSeed(seed);
      ann2.resetSeed(seed);
      ann3.resetSeed(seed);
      ann4.resetSeed(seed);
      
      for (int i = 0; i < nY; ++i) {
        float* q = Y + i*d;
        ann1.query(d, q, &Z, &S);
        REQUIRE(Z == mu1[i]);
        REQUIRE(S == (k >= nX ? nX : k > 0 ? nX + m : m));
        
        ann2.query(d, q, &Z, &S);
        REQUIRE(Z == mu2[i]);
        REQUIRE(S == (k >= nX ? nX : k > 0 ? nX + m : m));
        
        ann3.query(d, q, &Z, &S);
        REQUIRE(Z == mu1[i]);
        REQUIRE(S == (k > 0 ? -1 : m));
        
        ann4.query(d, q, &Z, &S);
        REQUIRE(Z == mu2[i]);
        REQUIRE(S == (k > 0 ? -1 : m));
      }

      ann1.resetSeed(seed);
      ann2.resetSeed(seed);
      ann3.resetSeed(seed);
      ann4.resetSeed(seed);

      for (int i = 0; i < nY; ++i) {
        float* q = Y + i*d;
        ann1.query(d, q, &Z, nullptr);
        REQUIRE(Z == mu1[i]);
        
        ann2.query(d, q, &Z, nullptr);
        REQUIRE(Z == mu2[i]);
        
        ann3.query(d, q, &Z, nullptr);
        REQUIRE(Z == mu1[i]);
        
        ann4.query(d, q, &Z, nullptr);
        REQUIRE(Z == mu2[i]);
      }
    }
  }
  allocator.free(data);
}




TEST_CASE( "test_ann_estimator_permuted1", "[annestimatorpermuted]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  LinearScan<float> ls(Metric::EUCLIDEAN);
  ls.fit(nX, d, X);

  float h = 25;

  KdeEstimator::PseudoRandomNumberGenerator::ValueType seed = 0xe492577c;

  vector<float> mu(nY);
  vector<float> Z(nY);
  vector<int> samples(nY);

  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (0, Kernel::EXPONENTIAL, 0, 0, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (-1, Kernel::EXPONENTIAL, 0, 0, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (h, Kernel::LAPLACIAN, 0, 0, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (h, Kernel::EXPONENTIAL, -1, 0, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (h, Kernel::EXPONENTIAL, 0, -1, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (h, Kernel::EXPONENTIAL, 0, 0, &ls).query(nY, d, Y, &Z[0])),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (h, Kernel::EXPONENTIAL, nX+1, 0, &ls).fit(nX, d, X)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (h, Kernel::EXPONENTIAL, 0, nX+1, &ls).fit(nX, d, X)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<float,LinearScan<float>>
                     (h, Kernel::EXPONENTIAL, 501, nX-500, &ls).fit(nX, d, X)),
                    invalid_argument);

  AnnEstimatorPermuted<float,LinearScan<float>> aep(h, Kernel::EXPONENTIAL, 0, 0, &ls, seed);
  aep.fit(nX, d, X);
  REQUIRE_THROWS_AS(aep.query(d-1, X, &Z[0]), invalid_argument);
  
  vector<uint32_t> indices(nX);
  for (int i = 0; i < nX; ++i)
    indices[i] = i;
  std::mt19937 rng(seed);
  std::shuffle(indices.begin(), indices.end(), rng);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
  REQUIRE(array::equal(nX, &indices[0], aep.getXpermutedIdx()));
  const float* Xp = aep.getXpermuted();
  for (int i = 0; i < nX; ++i) {
    REQUIRE(array::equal(d, X + indices[i]*d, Xp + i*d));
  }
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

  aep.query(nY, d, Y, &Z[0], &samples[0]);
  REQUIRE(array::allEqual(d, &Z[0], 0.0f));
  REQUIRE(array::allEqual(d, &samples[0], 0));
  
  RandomSamplingPermuted rsp(h, Kernel::EXPONENTIAL, 1, seed);
  rsp.fit(nX, d, X);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
  const float* Xp2 = rsp.getXpermuted();
  REQUIRE(array::equal(nX*d, Xp, Xp2));
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

  const float DELTA = 1e-6f;
  
  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN }) {
    const int m = 500;
    aep = AnnEstimatorPermuted<float,LinearScan<float>>(h, kernel, 0, m, &ls, seed);
    aep.fit(nX, d, X);
    rsp = RandomSamplingPermuted(h, kernel, m, seed);
    rsp.fit(nX, d, X);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nX*d, aep.getXpermuted(), rsp.getXpermuted()));
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    aep.query(nY, d, Y, &Z[0], &samples[0]);
    rsp.query(nY, d, Y, &mu[0]);

    for (int i = 0; i < nY; ++i) {
      REQUIRE(std::abs(mu[i] - Z[i]) < DELTA);
    }
    
    REQUIRE(array::allEqual(nY, &samples[0], m));    
    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));

    aep = AnnEstimatorPermuted<float,LinearScan<float>>(h, kernel, nX, 0, &ls, seed);
    aep.fit(nX, d, X);
    NaiveKde<float> nkde(h, kernel);
    nkde.fit(nX, d, X);
    aep.query(nY, d, Y, &Z[0], &samples[0]);
    nkde.query(nY, d, Y, &mu[0]);

    for (int i = 0; i < nY; ++i) {
      REQUIRE(std::abs(mu[i] - Z[i]) < DELTA);
    }

    REQUIRE(array::allEqual(nY, &samples[0], nX));    
    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));

    const int k = 1000;
    aep = AnnEstimatorPermuted<float,LinearScan<float>>(h, kernel, k, m, &ls, seed);
    aep.fit(nX, d, X);

    auto kfun = [&](float dist) {
      if (kernel == Kernel::EXPONENTIAL)
        return std::exp(-dist/h);
      else
        return std::exp(-dist*dist/h/h/2);
    };
    vector<int> nns(k);
    vector<float> dists(k);
    vector<float> scratch(d);
    int idx = 0;
    for (int i = 0; i < nY; ++i) {
      float* q = Y + i*d;
      ls.query(d, k, q, &nns[0], &dists[0]);
      float Z1 = 0;
      for (auto dist : dists)
        Z1 += kfun(dist);

      float Z2 = 0;
      int rsCount = 0;
      int rsSampleCount = 0;
      int correctionCount = 0;
      float correctionAmount = 0;
      while (rsCount < m) {
        int id = indices[idx];
        float* x = X + id*d;
        float dist = array::euclideanDistance(d, q, x, &scratch[0]);
        if (std::find(nns.begin(), nns.end(), id) == nns.end()) {
          Z2 += kfun(dist);
          ++rsCount;
        }
        else {
          correctionAmount += kfun(dist);
          ++correctionCount;
        }
        ++rsSampleCount;
        idx = (idx + 1) % nX;
      }

      aep.query(d, q, &Z[i], &samples[i]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(aep.getSampleIdx() == idx);
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(samples[i] == nX + rsSampleCount);
      float Z3 = Z1 / nX + Z2 * (nX-k) / m / nX;
      REQUIRE(std::abs(Z[i] - Z3) < DELTA);
    }
    vector<float> Z2(nY);
    vector<int> samples2(nY);
    aep = AnnEstimatorPermuted<float,LinearScan<float>>(h, kernel, k, m, &ls, seed);
    aep.fit(nX, d, X);
    aep.query(nY, d, Y, &Z2[0], &samples2[0]);
    REQUIRE(array::equal(nY, &samples[0], &samples2[0]));
    REQUIRE(array::equal(nY, &Z[0], &Z2[0]));

    aep = AnnEstimatorPermuted<float,LinearScan<float>>(h, kernel, k, m, &ls, seed + 1);
    aep.fit(nX, d, X);
    aep.query(nY, d, Y, &Z[0], &samples[0]);
    AnnEstimatorPermuted<float,LinearScan<float>> aep2(h, kernel, k, m, &ls, seed + 2);
    aep2.fit(nX, d, X);
    aep2.query(nY, d, Y, &Z2[0], &samples2[0]);
    REQUIRE(!array::equal(nY, &samples[0], &samples2[0]));
    REQUIRE(!array::equal(nY, &Z[0], &Z2[0]));
    for (int i = 0; i < nY; ++i) {
      REQUIRE(nX + m <= samples[i]);
      REQUIRE(samples[i] <= 2*nX);
      REQUIRE(nX + m <= samples2[i]);
      REQUIRE(samples2[i] <= 2*nX);
    }
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(!array::equal(nX*d, aep.getXpermuted(), aep2.getXpermuted()));
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    aep2.resetSeed(seed + 1);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nX*d, aep.getXpermuted(), aep2.getXpermuted()));
    REQUIRE(aep.getSampleIdx() != aep2.getSampleIdx());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

    aep2 = AnnEstimatorPermuted<float,LinearScan<float>>(h, kernel, k, m, &ls, seed + 1);
    aep2.fit(nX, d, X);
    aep2.query(nY, d, Y, &Z2[0], &samples2[0]);
    REQUIRE(array::equal(nY, &samples[0], &samples2[0]));
    for (int i = 0; i < nY; ++i) {
      // these should be equal but are not because of some MKL weirdness
      // REQUIRE(Z[i] == Z2[i]);
      REQUIRE(std::abs(Z[i] - Z2[i]) < 1e-7f);
    }
    // REQUIRE(array::equal(nY, &Z[0], &Z2[0]));
    REQUIRE(array::almostEqual(nY, &Z[0], &Z2[0], 1e-7f));
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nX*d, aep.getXpermuted(), aep2.getXpermuted()));
    REQUIRE(aep.getSampleIdx() == aep2.getSampleIdx());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

    AnnEstimatorPermuted<float,LinearScan<float>> aep3(h, kernel, k, m, &ls, seed + 3);
    aep3.fit(nX, d, X);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(!array::equal(nX*d, aep.getXpermuted(), aep3.getXpermuted()));
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    aep3.resetSeed(seed + 1);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nX*d, aep.getXpermuted(), aep3.getXpermuted()));
    REQUIRE(aep3.getSampleIdx() == 0);
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    aep3.query(nY, d, Y, &Z2[0], &samples2[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(aep.getSampleIdx() == aep3.getSampleIdx());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nY, &samples[0], &samples2[0]));
    // REQUIRE(array::equal(nY, &Z[0], &Z2[0]));
    REQUIRE(array::almostEqual(nY, &Z[0], &Z2[0], 1e-7f));
  }
  allocator.free(data);
}


TEST_CASE( "test_ann_estimator_permuted2", "[annestimatorpermuted]" ) {
  double* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  double *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  LinearScan<double> ls(Metric::EUCLIDEAN);
  ls.fit(nX, d, X);

  double h = 25;

  KdeEstimator::PseudoRandomNumberGenerator::ValueType seed = 0xd584f978;

  vector<double> mu(nY);
  vector<double> Z(nY);
  vector<int> samples(nY);

  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (0, Kernel::EXPONENTIAL, 0, 0, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (-1, Kernel::EXPONENTIAL, 0, 0, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (h, Kernel::LAPLACIAN, 0, 0, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (h, Kernel::EXPONENTIAL, -1, 0, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (h, Kernel::EXPONENTIAL, 0, -1, &ls)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (h, Kernel::EXPONENTIAL, 0, 0, &ls).query(nY, d, Y, &Z[0])),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (h, Kernel::EXPONENTIAL, nX+1, 0, &ls).fit(nX, d, X)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (h, Kernel::EXPONENTIAL, 0, nX+1, &ls).fit(nX, d, X)),
                    invalid_argument);
  REQUIRE_THROWS_AS((AnnEstimatorPermuted<double,LinearScan<double>>
                     (h, Kernel::EXPONENTIAL, 501, nX-500, &ls).fit(nX, d, X)),
                    invalid_argument);

  AnnEstimatorPermuted<double,LinearScan<double>> aep(h, Kernel::EXPONENTIAL, 0, 0, &ls, seed);
  aep.fit(nX, d, X);
  REQUIRE_THROWS_AS(aep.query(d-1, X, &Z[0]), invalid_argument);
  
  vector<uint32_t> indices(nX);
  for (int i = 0; i < nX; ++i)
    indices[i] = i;
  std::mt19937 rng(seed);
  std::shuffle(indices.begin(), indices.end(), rng);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
  REQUIRE(array::equal(nX, &indices[0], aep.getXpermutedIdx()));
  const double* Xp = aep.getXpermuted();
  for (int i = 0; i < nX; ++i) {
    REQUIRE(array::equal(d, X + indices[i]*d, Xp + i*d));
  }
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

  aep.query(nY, d, Y, &Z[0], &samples[0]);
  REQUIRE(array::allEqual(d, &Z[0], 0.0));
  REQUIRE(array::allEqual(d, &samples[0], 0));
  
  RandomSamplingPermuted rsp(h, Kernel::EXPONENTIAL, 1, seed);
  rsp.fit(nX, d, X);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
  const double* Xp2 = rsp.getXpermuted();
  REQUIRE(array::equal(nX*d, Xp, Xp2));
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

  const double DELTA = 1e-6f;
  
  for (Kernel kernel : { Kernel::EXPONENTIAL, Kernel::GAUSSIAN }) {
    const int m = 500;
    aep = AnnEstimatorPermuted<double,LinearScan<double>>(h, kernel, 0, m, &ls, seed);
    aep.fit(nX, d, X);
    rsp = RandomSamplingPermuted(h, kernel, m, seed);
    rsp.fit(nX, d, X);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nX*d, aep.getXpermuted(), rsp.getXpermuted()));
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    aep.query(nY, d, Y, &Z[0], &samples[0]);
    rsp.query(nY, d, Y, &mu[0]);

    for (int i = 0; i < nY; ++i) {
      REQUIRE(std::abs(mu[i] - Z[i]) < DELTA);
    }
    
    REQUIRE(array::allEqual(nY, &samples[0], m));    
    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));

    aep = AnnEstimatorPermuted<double,LinearScan<double>>(h, kernel, nX, 0, &ls, seed);
    aep.fit(nX, d, X);
    NaiveKde<double> nkde(h, kernel);
    nkde.fit(nX, d, X);
    aep.query(nY, d, Y, &Z[0], &samples[0]);
    nkde.query(nY, d, Y, &mu[0]);

    for (int i = 0; i < nY; ++i) {
      REQUIRE(std::abs(mu[i] - Z[i]) < DELTA);
    }

    REQUIRE(array::allEqual(nY, &samples[0], nX));    
    REQUIRE(array::almostEqual(nY, &mu[0], &Z[0], DELTA));

    const int k = 1000;
    aep = AnnEstimatorPermuted<double,LinearScan<double>>(h, kernel, k, m, &ls, seed);
    aep.fit(nX, d, X);

    auto kfun = [&](double dist) {
      if (kernel == Kernel::EXPONENTIAL)
        return std::exp(-dist/h);
      else
        return std::exp(-dist*dist/h/h/2);
    };
    vector<int> nns(k);
    vector<double> dists(k);
    vector<double> scratch(d);
    int idx = 0;
    for (int i = 0; i < nY; ++i) {
      double* q = Y + i*d;
      ls.query(d, k, q, &nns[0], &dists[0]);
      double Z1 = 0;
      for (auto dist : dists)
        Z1 += kfun(dist);

      double Z2 = 0;
      int rsCount = 0;
      int rsSampleCount = 0;
      int correctionCount = 0;
      double correctionAmount = 0;
      while (rsCount < m) {
        int id = indices[idx];
        double* x = X + id*d;
        double dist = array::euclideanDistance(d, q, x, &scratch[0]);
        if (std::find(nns.begin(), nns.end(), id) == nns.end()) {
          Z2 += kfun(dist);
          ++rsCount;
        }
        else {
          correctionAmount += kfun(dist);
          ++correctionCount;
        }
        ++rsSampleCount;
        idx = (idx + 1) % nX;
      }

      aep.query(d, q, &Z[i], &samples[i]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(aep.getSampleIdx() == idx);
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
      REQUIRE(samples[i] == nX + rsSampleCount);
      double Z3 = Z1 / nX + Z2 * (nX-k) / m / nX;
      REQUIRE(std::abs(Z[i] - Z3) < DELTA);
    }
    vector<double> Z2(nY);
    vector<int> samples2(nY);
    aep = AnnEstimatorPermuted<double,LinearScan<double>>(h, kernel, k, m, &ls, seed);
    aep.fit(nX, d, X);
    aep.query(nY, d, Y, &Z2[0], &samples2[0]);
    REQUIRE(array::equal(nY, &samples[0], &samples2[0]));
    REQUIRE(array::equal(nY, &Z[0], &Z2[0]));
    
    ///

    aep = AnnEstimatorPermuted<double,LinearScan<double>>(h, kernel, k, m, &ls, seed + 1);
    aep.fit(nX, d, X);
    aep.query(nY, d, Y, &Z[0], &samples[0]);
    AnnEstimatorPermuted<double,LinearScan<double>> aep2(h, kernel, k, m, &ls, seed + 2);
    aep2.fit(nX, d, X);
    aep2.query(nY, d, Y, &Z2[0], &samples2[0]);
    REQUIRE(!array::equal(nY, &samples[0], &samples2[0]));
    REQUIRE(!array::equal(nY, &Z[0], &Z2[0]));
    for (int i = 0; i < nY; ++i) {
      REQUIRE(nX + m <= samples[i]);
      REQUIRE(samples[i] <= 2*nX);
      REQUIRE(nX + m <= samples2[i]);
      REQUIRE(samples2[i] <= 2*nX);
    }
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(!array::equal(nX*d, aep.getXpermuted(), aep2.getXpermuted()));
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    aep2.resetSeed(seed + 1);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nX*d, aep.getXpermuted(), aep2.getXpermuted()));
    REQUIRE(aep.getSampleIdx() != aep2.getSampleIdx());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

    aep2 = AnnEstimatorPermuted<double,LinearScan<double>>(h, kernel, k, m, &ls, seed + 1);
    aep2.fit(nX, d, X);
    aep2.query(nY, d, Y, &Z2[0], &samples2[0]);
    REQUIRE(array::equal(nY, &samples[0], &samples2[0]));
    REQUIRE(array::equal(nY, &Z[0], &Z2[0]));
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nX*d, aep.getXpermuted(), aep2.getXpermuted()));
    REQUIRE(aep.getSampleIdx() == aep2.getSampleIdx());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

    AnnEstimatorPermuted<double,LinearScan<double>> aep3(h, kernel, k, m, &ls, seed + 3);
    aep3.fit(nX, d, X);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(!array::equal(nX*d, aep.getXpermuted(), aep3.getXpermuted()));
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    aep3.resetSeed(seed + 1);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nX*d, aep.getXpermuted(), aep3.getXpermuted()));
    REQUIRE(aep3.getSampleIdx() == 0);
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    aep3.query(nY, d, Y, &Z2[0], &samples2[0]);
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(aep.getSampleIdx() == aep3.getSampleIdx());
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    REQUIRE(array::equal(nY, &samples[0], &samples2[0]));
    REQUIRE(array::equal(nY, &Z[0], &Z2[0]));
  }
  allocator.free(data);
}



TEST_CASE( "test_vml_mode", "[vmlmode]" ) {
  float* data;
  int n, d;
  constructMediumTestSet(&data, &n, &d);
  float *X, *Y;
  int nX, nY;
  array::splitInHalf(n, d, &nX, &nY, data, &X, &Y);

  vector<float> mu(nY);

  unsigned int oldMode = vmlSetMode(VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);

  float h1 = 100;
  float h2 = 0.1;

  NaiveKde<float> nkde1(h1, Kernel::EXPONENTIAL);
  nkde1.fit(nX, d, X);
  auto start = std::chrono::steady_clock::now();
  nkde1.query(nY, d, Y, &mu[0]);
  auto end = std::chrono::steady_clock::now();
  std::chrono::nanoseconds diff = end - start;
  float t1 = diff.count();

  NaiveKde<float> nkde2(h2, Kernel::EXPONENTIAL);
  nkde2.fit(nX, d, X);
  start = std::chrono::steady_clock::now();
  nkde2.query(nY, d, Y, &mu[0]);
  end = std::chrono::steady_clock::now();
  diff = end - start;
  float t2 = diff.count();

  REQUIRE(t2/t1 > 10);

  vmlSetMode(VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_NOERR);

  NaiveKde<float> nkde3(h2, Kernel::EXPONENTIAL);
  nkde3.fit(nX, d, X);
  start = std::chrono::steady_clock::now();
  nkde3.query(nY, d, Y, &mu[0]);
  end = std::chrono::steady_clock::now();
  diff = end - start;
  t2 = diff.count();

  REQUIRE(std::abs(t2-t1)/t1 < 0.5);
    
  vmlSetMode(oldMode);
  allocator.free(data);
}



int main(int argc, char* argv[]) {
  mkl_set_num_threads(1);
  vmlSetMode(VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_NOERR);
  return Catch::Session().run(argc, argv);
}

