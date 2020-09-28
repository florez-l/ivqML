// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "NeuralNetwork.h"




#include <iostream>





// -------------------------------------------------------------------------
template< class _TScalar >
NeuralNetwork< _TScalar >::
NeuralNetwork( )
{
}

// -------------------------------------------------------------------------
template< class _TScalar >
NeuralNetwork< _TScalar >::
NeuralNetwork( const Self& other )
{
  /* TODO
     this->m_Layers.clear( );
     this->m_Layers.insert(
     this->m_Layers.begin( ), other.m_Layers.begin( ), other.m_Layers.end( )
     );
  */
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
Self& NeuralNetwork< _TScalar >::
operator=( const Self& other )
{
  /* TODO
     this->m_Layers.clear( );
     this->m_Layers.insert(
     this->m_Layers.begin( ), other.m_Layers.begin( ), other.m_Layers.end( )
     );
  */
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
  assert( this->m_Layers.size( ) > 0 );
  this->add( this->m_Layers.rbegin( )->second.output_size( ), o, f );
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
  /* TODO
     if( this->m_Layers.size( ) > 0 )
     assert( l.input_size( ) == this->m_Layers.back( ).output_size( ) );
  */
  this->m_Layers[ this->m_Layers.size( ) + 1 ] = l;
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
init( bool randomly )
{
  for( auto& l: this->m_Layers )
    l.second.init( randomly );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
TColVector NeuralNetwork< _TScalar >::
operator()( const TColVector& x ) const
{
  assert( this->m_Layers.size( ) > 1 );

  auto lIt = this->m_Layers.begin( );
  TColVector z = lIt->second.operator()( x.transpose( ) );
  for( lIt++; lIt != this->m_Layers.end( ); ++lIt )
    z = lIt->second.operator()( z );
  return( z );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
TScalar NeuralNetwork< _TScalar >::
cost(
  std::vector< TMatrix >& dw, std::vector< TMatrix >& db,
  const TMatrix& X, const TMatrix& Y, const TScalar& lambda
  )
{
  // Intermediary values
  using _TCols = std::map< unsigned int, TColVector >;
  using _TMatrices = std::map< unsigned int, TMatrix >;
  _TCols a, z, d;
  _TMatrices D;

  // Initialize fwd propagation
  typename TLayers::iterator fIt = this->m_Layers.begin( );
  a[ 0 ] = TColVector::Ones( fIt->second.input_size( ) );
  for( ; fIt != this->m_Layers.end( ); ++fIt )
  {
    a[ fIt->first ] = TColVector::Ones( fIt->second.output_size( ) );
    z[ fIt->first ] = TColVector::Zero( fIt->second.output_size( ) );
    D[ fIt->first ] =
      TMatrix::Zero( fIt->second.output_size( ), fIt->second.input_size( ) );
  } // end while

  // Initialize bck propagation
  typename TLayers::reverse_iterator bIt = this->m_Layers.rbegin( );
  d[ this->m_Layers.size( ) ] = TColVector::Zero( bIt->second.output_size( ) );
  for( bIt++; bIt != this->m_Layers.rend( ); ++bIt )
    d[ bIt->first ] = TColVector::Zero( bIt->second.output_size( ) );

  // Main loop
  TScalar J = TScalar( 0 );
  for( unsigned long i = 0; i < X.rows( ); ++i )
  {
    // a[ 0 ]
    typename _TCols::iterator afIt = a.begin( );
    afIt->second = X.row( i ).transpose( );

    // Fwd propagation
    fIt = this->m_Layers.begin( );
    typename _TCols::iterator zfIt = z.begin( );
    for( ; fIt != this->m_Layers.end( ); ++fIt, ++zfIt )
    {
      zfIt->second = fIt->second.linear_fwd( afIt->second );
      afIt++;
      afIt->second = fIt->second.sigma_fwd( zfIt->second );
    } // end for

    // Update cost
    auto y = ( Y.row( i ).array( ) == TScalar( 1 ) ).template cast< TScalar >( );
    J -= ( Eigen::log( afIt->second.array( ) ) * y ).sum( );
    J -= ( Eigen::log( TScalar( 1 ) - afIt->second.array( ) ) * ( TScalar( 1 ) - y ) ).sum( );

    // delta_out
    typename _TCols::reverse_iterator dbIt = d.rbegin( );
    dbIt->second = afIt->second - Y.row( i ).transpose( );

    // Bck propagation
    typename _TCols::reverse_iterator ndbIt = d.rbegin( );
    typename _TCols::reverse_iterator zbIt = z.rbegin( );
    bIt = this->m_Layers.rbegin( );
    for( dbIt++, zbIt++; dbIt != d.rend( ); ++dbIt, ++ndbIt, ++bIt, ++zbIt )
      dbIt->second = bIt->second.delta_bck( ndbIt->second, zbIt->second );

    // Gradient assembly
    typename _TMatrices::iterator DfIt = D.begin( );
    typename _TCols::iterator dfIt = d.begin( );
    afIt = a.begin( );
    for( ; dfIt != d.end( ); ++DfIt, ++afIt, ++dfIt )
      DfIt->second += dfIt->second * afIt->second.transpose( );

  } // end for

  dw.clear( );
  db.clear( );
  typename _TMatrices::iterator DfIt = D.begin( );
  typename _TCols::iterator dfIt = d.begin( );
  fIt = this->m_Layers.begin( );
  for( ; DfIt != D.end( ); ++DfIt, ++dfIt, ++fIt )
  {
    dw.push_back( DfIt->second / TScalar( X.rows( ) ) );
    if( lambda != TScalar( 0 ) )
      dw.back( ) += fIt->second.weights( ) * lambda;
    db.push_back( dfIt->second / TScalar( X.rows( ) ) );
  } // end for

  // Regularize cost and finish
  J /= TScalar( X.rows( ) );
  if( lambda != TScalar( 0 ) )
  {
    TScalar r = TScalar( 0 );
    for( const auto& l: this->m_Layers )
      r += l.second.regularization( );
    J += lambda * r / TScalar( X.rows( ) );
  } // end if
  return( J );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
train(
  const TMatrix& X, const TMatrix& Y,
  const TScalar& alpha, const TScalar& lambda,
  const TScalar& epsilon,
  std::ostream* os
  )
{
  // Initialization
  std::vector< TMatrix > dw, db;
  TScalar J = this->cost( dw, db, X, Y, lambda );
  TScalar dJ = J;

  // Main loop
  TScalar m = TScalar( X.rows( ) );
  bool stop = false;
  unsigned long nIter = 0;
  while( !stop )
  {
    // Update parameters
    typename TLayers::iterator lIt = this->m_Layers.begin( );
    typename std::vector< TMatrix >::const_iterator wIt = dw.begin( );
    typename std::vector< TMatrix >::const_iterator bIt = db.begin( );
    for( ; lIt != this->m_Layers.end( ); ++lIt, ++wIt, ++bIt )
    {
      lIt->second.weights( ) -= *wIt * alpha;
      lIt->second.biases( ) -= *bIt * alpha;
    } // end for

    // Update cost
    TScalar Jn = this->cost( dw, db, X, Y, lambda );
    dJ = J - Jn;
    if( dJ <= epsilon )
      stop = true;
    if( nIter % 1000 == 0 && os != nullptr )
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
      << "** Epsilon  : " << epsilon << std::endl
      << "********************************************" << std::endl;
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
  o << this->m_Layers.size( ) << std::endl;
  for( const auto& l: this->m_Layers )
    o << l.second << std::endl;
}

// -------------------------------------------------------------------------
template class NeuralNetwork< float >;
template class NeuralNetwork< double >;
template class NeuralNetwork< long double >;

// eof - $RCSfile$
