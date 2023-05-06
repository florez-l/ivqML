// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizers__Base__hxx__
#define __PUJ_ML__Optimizers__Base__hxx__

#include <cmath>
#include <limits>

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
PUJ_ML::Optimizers::Base< _C, _X, _Y >::
Base( TModel& m, const TX& X, const TY& Y )
  : m_Model( &m ),
    m_X( &X ),
    m_Y( &Y )
{
  this->m_Epsilon =
    std::pow(
      TReal( 10 ),
      std::floor(
        std::log10( std::numeric_limits< TReal >::epsilon( ) ) * TReal( 0.5 )
        )
      );

  this->set_debug(
    [](
      const TReal& J,
      const TReal& nG,
      const unsigned long long& epoch
      ) -> bool
    {
      return( false );
    }
    );
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::Base< _C, _X, _Y >::
set_debug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::Base< _C, _X, _Y >::
set_regularization_type_to_ridge( )
{
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::Base< _C, _X, _Y >::
set_regularization_type_to_LASSO( )
{
}

#endif // __PUJ_ML__Optimizers__Base__hxx__

// eof - $RCSfile$
