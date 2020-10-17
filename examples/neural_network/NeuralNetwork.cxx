// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "NeuralNetwork.h"


#include <iostream>

// -------------------------------------------------------------------------
template< class _TScalar >
NeuralNetwork< _TScalar >::
NeuralNetwork( const TScalar& epsilon )
  : m_Epsilon( epsilon )
{
}

// -------------------------------------------------------------------------
template< class _TScalar >
NeuralNetwork< _TScalar >::
NeuralNetwork( const Self& other )
{
  this->m_L.clear( );
  this->m_L.insert( this->m_L.begin( ), other.m_L.begin( ), other.m_L.end( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
Self& NeuralNetwork< _TScalar >::
operator=( const Self& other )
{
  this->m_L.clear( );
  this->m_L.insert( this->m_L.begin( ), other.m_L.begin( ), other.m_L.end( ) );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
add( unsigned int i, unsigned int o, const TActivation& f )
{
  this->add( TLayer( i, o, f ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
add( unsigned int o, const TActivation& f )
{
  assert( this->m_L.size( ) > 0 );
  this->add( this->m_L.back( ).output_size( ), o, f );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
add( const TMatrix& w, const TColVector& b, const TActivation& f )
{
  this->add( TLayer( w, b, f ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
add( const TLayer& l )
{
  if( this->m_L.size( ) > 0 )
    assert( l.input_size( ) == this->m_L.back( ).output_size( ) );
  this->m_L.push_back( l );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
init( bool randomly )
{
  for( TLayer& l: this->m_L )
    l.init( randomly );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
TMatrix NeuralNetwork< _TScalar >::
operator()( const TMatrix& x ) const
{
  typename TLayers::const_iterator lIt = this->m_L.begin( );
  TMatrix z = lIt->operator()( x );
  for( lIt++; lIt != this->m_L.end( ); ++lIt )
    z = lIt->operator()( z );
  return( z );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
TScalar NeuralNetwork< _TScalar >::
cost( const TMatrix& X, const TMatrix& Y ) const
{
  static const TScalar _1 = TScalar( 1 );

  TMatrix Yr = this->operator()( X.transpose( ) ).transpose( );
  auto y = ( Y.array( ) == _1 ).template cast< TScalar >( );
  TScalar J = -( Eigen::log( Yr.array( ) ) * y ).sum( );
  J -= ( Eigen::log( _1 - Yr.array( ) ) * ( _1 - y ) ).sum( );
  J /= TScalar( X.rows( ) );
  return( J );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
train(
  const TMatrix& X, const TMatrix& Y,
  const TScalar& alpha, const TScalar& lambda,
  std::ostream* os
  )
{
  // Initialize temporary values
  unsigned long L = this->m_L.size( );
  std::vector< TMatrix > dw( L );
  std::vector< TColVector > db( L ), a( L + 1 ), z( L ), d( L );
  a.shrink_to_fit( );
  z.shrink_to_fit( );
  d.shrink_to_fit( );
  dw.shrink_to_fit( );
  db.shrink_to_fit( );

  // First cost
  TScalar J = this->_cost_and_gradient( dw, db, a, z, d, X, Y );
  if( lambda != TScalar( 0 ) )
    for( unsigned long l = 0; l < L; ++l )
      J += this->m_L[ l ].regularization( ) * lambda;
  TScalar dJ = J;

  // Main loop
  bool stop = false;
  unsigned long nIter = 0;
  while( !stop )
  {
    // Update parameters
    for( unsigned long l = 0; l < L; ++l )
    {
      if( lambda != TScalar( 0 ) )
        this->m_L[ l ].weights( ) -=
          ( dw[ l ] * alpha ) +
          ( this->m_L[ l ].weights( ) * lambda );
      else
        this->m_L[ l ].weights( ) -= dw[ l ] * alpha;
      this->m_L[ l ].biases( ) -= db[ l ] * alpha;
    } // end for

    // Update cost
    TScalar Jn = this->_cost_and_gradient( dw, db, a, z, d, X, Y );
    if( lambda != TScalar( 0 ) )
      for( unsigned long l = 0; l < L; ++l )
        Jn += this->m_L[ l ].regularization( ) * lambda;
    dJ = J - Jn;
    if( dJ <= this->m_Epsilon )
      stop = true;
    if( nIter % 100 == 0 && os != nullptr )
      *os
        << "\33[2K\rIteration: " << nIter
        << "\tJ = " << J
        << "\tdJ = " << dJ
        << std::flush;
    J = Jn;
    nIter++;
  } // end while

  if( os != nullptr )
    *os
      << std::endl
      << "********************************************" << std::endl
      << "** ANN trained in " << nIter << " iterations" << std::endl
      << "** Final J  : " << J << std::endl
      << "** Final dJ : " << dJ << std::endl
      << "** Alpha    : " << alpha << std::endl
      << "** Lambda   : " << lambda << std::endl
      << "** Epsilon  : " << this->m_Epsilon << std::endl
      << "********************************************" << std::endl;
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
TMatrix NeuralNetwork< _TScalar >::
confusion_matrix( const TMatrix& X, const TMatrix& Y ) const
{
  TMatrix K = TMatrix::Zero( 2, 2 );
  auto R =
    ( this->operator()( X.transpose( ) ).array( ) >= 0.5 ).
    template cast< TScalar >( ).transpose( );
  auto RpY = Y.array( ) + R.array( );
  auto RmY = Y.array( ) - R.array( );
  K( 0, 0 ) = ( RpY == 0 ).template cast< TScalar >( ).sum( );
  K( 1, 1 ) = ( RpY == 2 ).template cast< TScalar >( ).sum( );
  K( 0, 1 ) = ( RmY < 0 ).template cast< TScalar >( ).sum( );
  K( 1, 0 ) = ( RmY > 0 ).template cast< TScalar >( ).sum( );
  return( K );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
TScalar NeuralNetwork< _TScalar >::
_cost_and_gradient(
  std::vector< TMatrix >& dw,
  std::vector< TColVector >& db,
  std::vector< TColVector >& a,
  std::vector< TColVector >& z,
  std::vector< TColVector >& d,
  const TMatrix& X, const TMatrix& Y
  ) const
{
  static const TScalar _0 = TScalar( 0 );
  static const TScalar _1 = TScalar( 1 );

  long L = this->m_L.size( );
  unsigned long m = X.rows( );
  TScalar sm = TScalar( m );
  TScalar J = _0;

  for( unsigned long i = 0; i < m; ++i )
  {
    // Fwd propagation
    a[ 0 ] = X.row( i ).transpose( );
    for( long l = 0; l < L; ++l )
    {
      z[ l ] = this->m_L[ l ].linear_fwd( a[ l ] );
      a[ l + 1 ] = this->m_L[ l ].sigma_fwd( z[ l ] );
    } // end for

    // Update cost
    auto y = ( Y.row( i ).array( ) == _1 ).template cast< TScalar >( );

    J -= ( Eigen::log( a[ L ].transpose( ).array( ) ) * y ).sum( ) / sm;
    J -= ( Eigen::log( _1 - a[ L ].transpose( ).array( ) ) * ( _1 - y ) ).sum( ) / sm;

    // Output error
    d[ L - 1 ] =
      ( a[ L ] - Y.row( i ).transpose( ) ).array( ) *
      this->m_L[ L - 1 ].sigma( )( z[ L - 1 ], true ).array( );
    if( i == 0 )
    {
      dw[ L - 1 ]  = ( d[ L - 1 ] * a[ L - 1 ].transpose( ) ) / sm;
      db[ L - 1 ]  = d[ L - 1 ] / sm;
    }
    else
    {
      dw[ L - 1 ] += ( d[ L - 1 ] * a[ L - 1 ].transpose( ) ) / sm;
      db[ L - 1 ] += d[ L - 1 ] / sm;
    } // end if

    // Bck propagation
    for( long l = L - 2; l >= 0; --l )
    {
      d[ l ] =
        ( this->m_L[ l + 1 ].weights( ).transpose( ) * d[ l + 1 ] ).array( ) *
        this->m_L[ l ].sigma( )( z[ l ], true ).array( );
      if( i == 0 )
      {
        dw[ l ]  = ( d[ l ] * a[ l ].transpose( ) ) / sm;
        db[ l ]  = d[ l ] / sm;
      }
      else
      {
        dw[ l ] += ( d[ l ] * a[ l ].transpose( ) ) / sm;
        db[ l ] += d[ l ] / sm;
      } // end if
    } // end for
  } // end for

  // Ok, we're done!
  return( J );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
_ReadFrom( std::istream& i )
{
  unsigned int N;
  i >> N;
  for( unsigned int n = 0; n < N; ++n )
  {
    TLayer l;
    i >> l;
    this->add( l );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
_CopyTo( std::ostream& o ) const
{
  o << this->m_L.size( ) << std::endl;
  for( const TLayer& l: this->m_L )
    o << l << std::endl;
}

// -------------------------------------------------------------------------
template class NeuralNetwork< float >;
template class NeuralNetwork< double >;
template class NeuralNetwork< long double >;

// eof - $RCSfile$
