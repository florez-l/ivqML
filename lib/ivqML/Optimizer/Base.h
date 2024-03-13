// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__h__
#define __ivqML__Optimizer__Base__h__

#include <functional>
#include <limits>
#include <string>
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
      using TScalar  = typename TCost::TScalar;
      using TNatural = typename TCost::TNatural;
      using TMat     = typename TCost::TMat;
      using TCol     = typename TCost::TCol;
      using TRow     = typename TCost::TRow;
      using TMatMap  = typename TCost::TMatMap;
      using TColMap  = typename TCost::TColMap;
      using TRowMap  = typename TCost::TRowMap;
      using TMatCMap = typename TCost::TMatCMap;
      using TColCMap = typename TCost::TColCMap;
      using TRowCMap = typename TCost::TRowCMap;

      using TDebugger = std::function< bool( const TScalar&, const TScalar&, const TModel*, const TNatural&, bool d ) >;

    public:
      ivqMLAttributeMacro( batch_size, TNatural, 0 );
      ivqMLAttributeMacro( epsilon, TScalar, 0 );
      ivqMLAttributeMacro( lambda, TScalar, 0 );
      ivqMLAttributeMacro( debug_iterations, TNatural, 100 );
      ivqMLAttributeMacro(
        max_iterations, TNatural, std::numeric_limits< TNatural >::max( )
        );

    public:
      Base( );
      virtual ~Base( );

      std::string parse_arguments( int c, char** v );

      bool has_model( ) const;
      TModel& model( );
      const TModel& model( ) const;

      template< class _TInputX, class _TInputY >
      void set_data(
        const Eigen::EigenBase< _TInputX >& iX,
        const Eigen::EigenBase< _TInputY >& iY
        );

      void set_debugger( TDebugger d );
      void unset_debugger( );

      virtual void fit( ) = 0;

    private:
      Base( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;

    protected:
      TModel* m_Model { nullptr };
      bool m_ManagedModel { false };

      std::vector< TCost > m_Costs;

      boost::program_options::options_description m_Options { "Options." };

      TDebugger m_Debugger;
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Base.hxx>

#endif // __ivqML__Optimizer__Base__h__

// eof - $RCSfile$
