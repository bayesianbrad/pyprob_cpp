#include <pyprob_cpp.h>
#include "xtensor/xadapt.hpp"

int main(int argc, char *argv[])
{
  printf("Sampling from Normal({1, 2, 3}, {0.1, 0.01, 0.001})");
  auto normal = pyprob_cpp::distributions::Normal(xt::xarray<double> {1, 2, 3}, xt::xarray<double> {0.1, 0.01, 0.001});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(normal) << std::endl;
  }
  pyprob_cpp::observe(normal, xt::xarray<double> {0, 1, 0});

  printf("\nSampling from Uniform(0, 1)");
  auto uniform = pyprob_cpp::distributions::Uniform(0, 1);
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(uniform) << std::endl;
  }
  pyprob_cpp::observe(uniform, xt::xarray<double> {0});

  printf("\nSampling from Categorical({0.2, 0.7, 0.1})");
  auto categorical = pyprob_cpp::distributions::Categorical(xt::xarray<double> {0.2, 0.7, 0.1});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(categorical) << std::endl;
  }
  pyprob_cpp::observe(categorical, 2);

  printf("\nSampling from Poisson({2, 10, 20})");
  auto poisson = pyprob_cpp::distributions::Poisson(xt::xarray<double> {2, 10, 20});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(poisson) << std::endl;
  }
  pyprob_cpp::observe(poisson, xt::xarray<double> {0, 1, 2});

//  printf("Sampling from Beta({0.1,0.2,0.85},{0.3,0.4,1})");
//  auto gamma = pyprob_cpp::distributions::Beta(xt::xarray<double> {0.1,0.2,0.85}, xt::xarray<double>{0.3,0.4,1});
//  for (int i = 0; i < 10; i++)
//  {
//    std::cout << pyprob_cpp::sample(beta) << std::endl;
//  }
//  pyprob_cpp::observe(beta, xt::xarray<double> {1,0.5,4.0});

  printf("Sampling from Gamma({1,2,3},{1.2,3.4,5.2})");
  auto gamma = pyprob_cpp::distributions::Gamma(xt::xarray<double> {1,2,3}, xt::xarray<double> {1.2,3.4,5.2});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(gamma) << std::endl;
  }
  pyprob_cpp::observe(gamma, xt::xarray<double> {2,1,6});

  printf("Sampling from LogNormal(({1, 2, 3}, {0.1, 0.01, 0.001})");
  auto log_normal = pyprob_cpp::distributions::LogNormal(xt::xarray<double> {1, 2, 3}, xt::xarray<double>  {0.1, 0.01, 0.001});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(log_normal) << std::endl;
  }
  pyprob_cpp::observe(log_normal, xt::xarray<double> {3,25,16});

  printf("\nSampling from Exponential({2, 6, 23})");
  auto exponential = pyprob_cpp::distributions::Exponential(xt::xarray<double> {2, 6, 23});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(exponential) << std::endl;
  }
  pyprob_cpp::observe(exponential, xt::xarray<double> {1, 2, 14});

  printf("\nSampling from Weibull({2, 6, 23}, {1,5,2})");
  auto weibull = pyprob_cpp::distributions::Weibull(xt::xarray<double> {2, 6, 23}, xt::xarray<double> {1,5,2});
  for (int i = 0; i < 10; i++)
  {
    std::cout << pyprob_cpp::sample(weibull) << std::endl;
  }
  pyprob_cpp::observe(weibull, xt::xarray<double> {1, 2, 14});

  return 0;

}
