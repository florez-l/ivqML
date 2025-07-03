// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Common/MeanShift.h>

int main( int argc, char** argv )
{
  using TMeanShift = ivqML::Common::MeanShift< double, unsigned long long >;
  using TReal = TMeanShift::TReal;
  using TNatural = TMeanShift::TNatural;

  TReal I[ ] =
    {
      1.0, 1.0,
      1.2, 1.1,
      1.0, 1.3,
      1.1, 0.9,
      5.0, 5.0,
      5.1, 5.2,
      5.3, 5.0,
      5.0, 5.1,
      0.5, 6.0,
      0.7, 6.1,
      0.6, 5.9
    };
  TNatural M = 11;
  TNatural N = 2;

  TMeanShift ms( I, N, M );
  std::vector< TReal > means;
  ms.GetMeans( std::back_inserter( means ) );

  std::cout << means.size( ) << std::endl;
  std::cout << "--------------------------------" << std::endl;
  for( const auto& v: means )
    std::cout << v << std::endl;
  
  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
