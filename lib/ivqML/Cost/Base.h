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
      using TScl     = typename TModel::TScl;
      using TNat     = typename TModel::TNat;
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
      Base( );
      virtual ~Base( );

      virtual void set_data(
        TScl* X, TScl* Y, const TNat& m, const TNat& n, const TNat& p
        );

      virtual TScl operator()(
        const TModel& model, TScl* G = nullptr
        ) const = 0;

    protected:
      TMatMap m_X { nullptr, 0, 0 };
      TMatMap m_Y { nullptr, 0, 0 };

      /* TODO
         mutable TScl* m_B { nullptr };
      */
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/Base.hxx>

#endif // __ivqML__Cost__Base__h__

// eof - $RCSfile$
