// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__hxx__
#define __ivqML__Optimizer__Base__hxx__

#include <random>

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Base< _TModel >::
Base( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Base< _TModel >::
~Base( )
{
  this->_free( );
}

// -------------------------------------------------------------------------
template< class _TModel >
typename ivqML::Optimizer::Base< _TModel >::
TDebugger ivqML::Optimizer::Base< _TModel >::
debugger( ) const
{
  return( this->m_Debugger );
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Base< _TModel >::
set_debugger( TDebugger d )
{
  this->m_Debugger = d;
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Base< _TModel >::
set_data( TReal* X, TReal* Y, const TNatural& M )
{
  this->_free( );
  this->m_X = X;
  this->m_Y = Y;
  this->m_M = M;
  this->m_OwnBuffer = false;
}

// -------------------------------------------------------------------------
template< class _TModel >
template< class _TBX, class _TBY >
void ivqML::Optimizer::Base< _TModel >::
set_data( Eigen::EigenBase< _TBX >& X, Eigen::EigenBase< _TBY >& Y )
{
  this->_free( );
  this->m_X
    =
    reinterpret_cast< TReal* >( std::calloc( X.size( ), sizeof( TReal ) ) );
  this->m_Y
    =
    reinterpret_cast< TReal* >( std::calloc( Y.size( ), sizeof( TReal ) ) );
  this->m_M = X.cols( );
  this->m_OwnBuffer = true;

  Eigen::Map< TMatrix >( this->m_X, X.rows( ), X.cols( ) )
    =
    X.derived( ).template cast< TReal >( );
  Eigen::Map< TMatrix >( this->m_Y, Y.rows( ), Y.cols( ) )
    =
    Y.derived( ).template cast< TReal >( );
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Base< _TModel >::
fit( TModel& model )
{
  TNatural* idx
    =
    reinterpret_cast< TNatural* >(
      std::calloc( this->m_M, sizeof( TNatural ) )
      );
  std::iota( idx, idx + this->m_M, 0 );

  TNatural bs = ( this->m_BatchSize == 0 )? this->m_M: this->m_BatchSize;
  bs = ( bs < this->m_M )? bs: this->m_M;

  TNatural last_batch_size = this->m_M % bs;
  TNatural n_batches = TNatural( this->m_M / bs );

  TBatches batches;
  TNatural* I = idx;
  for( TNatural b = 0; b < n_batches; ++b )
  {
    batches.push_back( TBatch( I, 1, bs ) );
    I += bs;
  } // end for
  if( last_batch_size > 0 )
    batches.push_back( TBatch( I, 1, last_batch_size ) );
  batches.shrink_to_fit( );

  std::random_device rnd_dev;
  std::mt19937 rnd_gen( rnd_dev( ) );
  this->_fit(
    model, batches,
    [ &idx, &rnd_gen, this ]( ) -> void
    {
      std::shuffle( idx, idx + this->m_M, rnd_gen );
    }
    );
  std::free( idx );
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Base< _TModel >::
_free( )
{
  if( this->m_OwnBuffer )
  {
    if( this->m_X != nullptr )
      std::free( this->m_X );
    if( this->m_Y != nullptr )
      std::free( this->m_Y );
  } // end if
  this->m_X = nullptr;
  this->m_Y = nullptr;
  this->m_M = 0;
  this->m_OwnBuffer = true;
}

#endif // __ivqML__Optimizer__Base__hxx__

// eof - $RCSfile$
