// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================


#include <iostream>




#include <PUJ/Model/Logistic.h>
#include <Eigen/Dense>
#include <cmath>
#include <random>

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::
Logistic( const TRow& w, const TScalar& b )
{
  this->m_Linear = new TLinear( w, b );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::
~Logistic( )
{
  delete this->m_Linear;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
unsigned long PUJ::Model::Logistic< _TScalar, _TTraits >::
GetDimensions( ) const
{
  return( this->m_Linear->GetDimensions( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TRow& PUJ::Model::Logistic< _TScalar, _TTraits >::
GetWeights( ) const
{
  return( this->m_Linear->GetWeights( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar& PUJ::Model::Logistic< _TScalar, _TTraits >::
GetBias( ) const
{
  return( this->m_Linear->GetBias( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Logistic< _TScalar, _TTraits >::
SetWeights( const TRow& w )
{
  this->m_Linear->SetWeights( w );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Logistic< _TScalar, _TTraits >::
SetBias( const TScalar& b )
{
  this->m_Linear->SetBias( b );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar PUJ::Model::Logistic< _TScalar, _TTraits >::
operator()( const TRow& x, bool threshold ) const
{
  TScalar z = this->m_Linear->operator()( x );
  if( z > TScalar( 40 ) )
    return( TScalar( 1 ) );
  else if( z < -TScalar( 40 ) )
    return( TScalar( 0 ) );
  else
  {
    TScalar a = TScalar( 1 ) / ( TScalar( 1 ) + std::exp( -z ) );
    if( threshold )
      return( TScalar( ( a < TScalar( 0.5 ) )? 0: 1 ) );
    else
      return( a );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TCol PUJ::Model::Logistic< _TScalar, _TTraits >::
operator()( const TMatrix& x, bool threshold ) const
{
  TCol z = this->m_Linear->operator()( x );
  return(
    z.unaryExpr(
      [=]( TScalar v ) -> TScalar
      {
        if( v > TScalar( 40 ) )
          return( TScalar( 1 ) );
        else if( v < -TScalar( 40 ) )
          return( TScalar( 0 ) );
        else
        {
          TScalar r = TScalar( 1 ) / ( TScalar( 1 ) + std::exp( -v ) );
          if( threshold )
            return( TScalar( ( r < TScalar( 0.5 ) )? 0: 1 ) );
          else
            return( r );
        } // end if
      }
      )
    );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::Cost::
Cost( const TMatrix& X, const TCol& y )
{
  this->m_Zeros.clear( );
  this->m_Ones.clear( );
  PUJ::visit_lambda(
    y,
    [&]( TScalar v, int i, int j )
    {
      if( v == 0 ) this->m_Zeros.push_back( i );
      else         this->m_Ones.push_back( i );
    }
    );

  /* TODO
     unsigned long long m =
     ( this->m_Zeros.size( ) < this->m_Ones.size( ) )?
     this->m_Zeros.size( ):
     this->m_Ones.size( );

     this->m_Zeros.resize( m );
     this->m_Ones.resize( m );

     std::vector< unsigned long long > all;
     all.insert( all.end( ), this->m_Zeros.begin( ), this->m_Zeros.end( ) );
     all.insert( all.end( ), this->m_Ones.begin( ), this->m_Ones.end( ) );

     auto rd = std::random_device {}; 
     auto rng = std::default_random_engine { rd() };
     std::shuffle( std::begin( all ), std::end( all ), rng );
  */
  this->m_X = X; // ( all, Eigen::placeholders::all );
  this->m_y = y; // ( all );

  this->m_Xy =
    ( this->m_X.array( ).colwise( ) * this->m_y.array( ) ).
    colwise( ).mean( );
  this->m_uy = this->m_y.mean( );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar PUJ::Model::Logistic< _TScalar, _TTraits >::Cost::
operator()( const TRow& t, TRow* g ) const
{
  static const TScalar eps = 1e-8; // std::numeric_limits< TScalar >::epsilon( );

  unsigned long long n = this->m_X.cols( );
  unsigned long long m = this->m_X.rows( );
  TCol a = Self( t.block( 0, 1, 1, n ), t( 0, 0 ) )( this->m_X, false );
  TScalar o = Eigen::log( a( this->m_Ones ).array( ) + eps ).sum( );
  TScalar z = Eigen::log( 1.0 - a( this->m_Zeros ).array( ) + eps ).sum( );

  if( g != nullptr )
  {
    if( g->cols( ) != n + 1 )
      *g = TRow::Zero( n + 1 );

    g->operator()( 0, 0 ) = a.mean( ) - this->m_uy;
    g->block( 0, 1, 1, n ) =
      ( this->m_X.array( ).colwise( ) * a.array( ) ).colwise( ).mean( ) -
      this->m_Xy.array( );

    /* TODO
       TMatrix lll( this->m_Ones.size( ), 2 );
       lll.block( 0, 0, this->m_Ones.size( ), 1 ) = a( this->m_Ones );
       lll.block( 0, 1, this->m_Ones.size( ), 1 ) = Eigen::log( a( this->m_Ones ).array( ) + eps );

       std::cout << lll << std::endl;
       std::cout << t << std::endl;
       std::cout << o << " " << z << " " << m << " : " << *g << std::endl;
       std::exit( 1 );
    */

  } // end if
  return( -( o + z ) / TScalar( m ) );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Model::Logistic< float >;
template class PUJ_ML_EXPORT PUJ::Model::Logistic< double >;
template class PUJ_ML_EXPORT PUJ::Model::Logistic< long double >;

// eof - $RCSfile$
