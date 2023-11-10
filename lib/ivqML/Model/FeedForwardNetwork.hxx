// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__FeedForwardNetwork__hxx__
#define __ivqML__Model__FeedForwardNetwork__hxx__


#include <iostream>





// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::FeedForwardNetwork< _S >::
operator()(
  Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
  bool derivative
  ) const
{
  TNatural L = this->number_of_layers( );
  std::vector< TMatrix > A( L + 1 );
  std::vector< TMatrix > Z( L );
  A.shrink_to_fit( );
  Z.shrink_to_fit( );

  A[ 0 ] = iX.derived( ).template cast< TScalar >( );
  for( TNatural l = 0; l < L; ++l )
  {
    Z[ l ] =
      ( A[ l ] * this->m_W[ l ] ).rowwise( )
      +
      this->m_B[ l ].row( 0 );
    A[ l + 1 ] = TMatrix::Zero( Z[ l ].rows( ), Z[ l ].cols( ) );
    this->m_F[ l ].second( A[ l + 1 ], Z[ l ], false );
  } // end for

  if( derivative )
  {
    // Backpropagation
    std::vector< TMatrix > D( L );
    D.shrink_to_fit( );

    if( this->m_IsLabeling )
      D[ L - 1 ] = A[ L ] - iY.derived( ).template cast< TScalar >( );
    /* TODO
       else // regression
    */
    for( TNatural j = 1; j < L; ++j )
    {
      TNatural l = L - j - 1;

      D[ l ] = TMatrix( Z[ l ].rows( ), Z[ l ].cols( ) );
      this->m_F[ l ].second( D[ l ], Z[ l ], true );
      auto B = this->m_W[ l + 1 ].transpose( ) * D[ l + 1 ];

      std::cout << "********************" << std::endl;
      std::cout << l << std::endl;
      std::cout << D[ l ].rows( ) << " " << D[ l ].cols( ) << std::endl;
      std::cout << B.rows( ) << " " << B.cols( ) << std::endl;
      std::cout << "********************" << std::endl;

    } // end for

    /* TODO:
       {
       D[ l ] = TMatrix( Z[ L - l - 1 ].rows( ), Z[ L - l - 1 ].cols( ) );
       this->m_F[ L - l - 1 ].second( D[ l ], Z[ L - l - 1 ], true );


       std::cout << "-----***-------" << std::endl << B << std::endl;


       D[ l ].array( ) *= B.array( );
       } // end for
    */

    for( const auto& d: D )
      std::cout << "------------" << std::endl << d << std::endl;
    std::exit( 1 );

  }
  else
    iY.derived( ) = A[ L ].template cast< typename _Y::Scalar >( );
}

#endif // __ivqML__Model__FeedForwardNetwork__hxx__

// eof - $RCSfile$
