// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__Base__h__
#define __ivqML__Cost__Base__h__

namespace ivqML
{
  namespace Cost
  {
    /**
     */
    template< class _TModel >
    class Base
    {
    public:
      using Self     = Base;
      using TModel   = _TModel;
      using TScalar  = typename TModel::TScalar;
      using TNatural = typename TModel::TNatural;
      using TMat     = typename TModel::TMat;
      using TCol     = typename TModel::TCol;
      using TRow     = typename TModel::TRow;
      using TMatMap  = typename TModel::TMatMap;
      using TColMap  = typename TModel::TColMap;
      using TRowMap  = typename TModel::TRowMap;
      using TMatCMap = typename TModel::TMatCMap;
      using TColCMap = typename TModel::TColCMap;
      using TRowCMap = typename TModel::TRowCMap;

    public:
      Base( _TModel& m );
      virtual ~Base( ) = default;

      template< class _TInputX, class _TInputY >
      void set_data(
        const Eigen::EigenBase< _TInputX >& iX,
        const Eigen::EigenBase< _TInputY >& iY
        );

      virtual TScalar operator()( TScalar* G = nullptr ) const = 0;

    protected:
      TModel* m_M;
      TMat m_X;
      TRow m_Y;
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/Base.hxx>

#endif // __ivqML__Cost__Base__h__

// eof - $RCSfile$
