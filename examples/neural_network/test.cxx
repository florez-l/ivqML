#include "ActivationFunctions.h"
#include <iostream>

int main( )
{
  using TFunc = ActivationFunctions::SoftPlus< long double >;

  TFunc f;
  TFunc::TColVector z = TFunc::TColVector::Random( 5 );

  std::cout << z << std::endl;
  std::cout << "-------------------" << std::endl;
  std::cout << f( z ) << std::endl;
  std::cout << "-------------------" << std::endl;
  std::cout << f[ z ] << std::endl;

  return( 0 );

}

// eof
