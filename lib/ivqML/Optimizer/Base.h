// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__h__
#define __ivqML__Optimizer__Base__h__

#include <functional>
#include <limits>
#include <vector>

#include <boost/program_options.hpp>

// -------------------------------------------------------------------------
#define ivqML_Optimizer_OptionMacro( _N, _O )                           \
  (                                                                     \
    _O,                                                                 \
    boost::program_options::value< decltype( this->m_##_N ) >           \
    ( &( this->m_##_N ) )->default_value( this->m_##_N ), ""            \
    )

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TCost >
    class Base
    {
    public:
      using Self     = Base;
      using TCost    = _TCost;
      using TModel   = typename TCost::TModel;
      using TScl     = typename TCost::TScl;
      using TNat     = typename TCost::TNat;
      using TMat     = typename TCost::TMat;
      using TCol     = typename TCost::TCol;
      using TRow     = typename TCost::TRow;
      using TMatMap  = typename TCost::TMatMap;
      using TColMap  = typename TCost::TColMap;
      using TRowMap  = typename TCost::TRowMap;
      using TMatCMap = typename TCost::TMatCMap;
      using TColCMap = typename TCost::TColCMap;
      using TRowCMap = typename TCost::TRowCMap;
      using TDebug   =
        std::function< bool( const TModel*, const TScl&, const TNat&, const TCost*, bool ) >;

    public:
      ivqMLAttributeMacro( batch_size, TNat, 0 );
      ivqMLAttributeMacro( epsilon, TScl, 0 );
      ivqMLAttributeMacro( lambda, TScl, 0 );
      ivqMLAttributeMacro( max_iter, TNat, std::numeric_limits< TNat >::max( ) );

    public:
      Base( );
      virtual ~Base( );

      virtual void register_options(
        boost::program_options::options_description& opt
        );

      template< class _TInputX, class _TInputY >
      void set_data(
        const Eigen::EigenBase< _TInputX >& iX,
        const Eigen::EigenBase< _TInputY >& iY
        );

      virtual void unset_debug( );
      virtual void fit( TModel& model ) = 0;

    private:
      Base( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;

    protected:

      /* TODO
         TMat m_X;
         TMat m_Y;
      */
      TScl* m_D { nullptr };
      TNat  m_M { 0 };
      TNat  m_N { 0 };
      TNat  m_P { 0 };

      TCost m_CostFromCompleteData;
      std::vector< TCost > m_Costs;

      TDebug m_Debug;
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Base.hxx>

#endif // __ivqML__Optimizer__Base__h__

// eof - $RCSfile$
