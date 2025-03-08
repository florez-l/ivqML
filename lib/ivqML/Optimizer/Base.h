// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__h__
#define __ivqML__Optimizer__Base__h__

#include <limits>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class Base
    {
    public:
      using TModel   = _TModel;
      using Self     = Base;
      using TReal    = typename TModel::TReal;
      using TNatural = typename TModel::TNatural;
      using TMat     = typename TModel::TMat;
      using TCol     = typename TModel::TCol;
      using TRow     = typename TModel::TRow;
      using TMatMap  = typename TModel::TMatMap;
      using TCMatMap = typename TModel::TCMatMap;
      using TColMap  = typename TModel::TColMap;
      using TCColMap = typename TModel::TCColMap;
      using TRowMap  = typename TModel::TRowMap;
      using TCRowMap = typename TModel::TCRowMap;

      using TDebug
      =
        std::function<
          bool( const TNatural&, const TReal&, const TReal&, const TReal& )
          >;

    public:
      Base( TModel& m );
      virtual ~Base( );

      TModel* model( ) const;
      const TReal& lambda1( ) const;
      const TReal& lambda2( ) const;
      const TReal& epsilon( ) const;

      void setLambda1( const TReal& l );
      void setLambda2( const TReal& l );
      void setEpsilon( const TReal& e );
      void setDebug( TDebug d );

    protected:
      TModel* m_Model { nullptr };

      TReal m_Lambda1 { 0 };
      TReal m_Lambda2 { 0 };
      TReal m_Epsilon { std::numeric_limits< TReal >::epsilon( ) };

      TDebug m_Debug
        {
          [](
            const TNatural& t,
            const TReal& nG,
            const TReal& J_tr, const TReal& J_te
            ) -> bool
          {
            return( false );
          }
        };
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Base.hxx>

#endif // __ivqML__Optimizer__Base__h__

// eof - $RCSfile$
