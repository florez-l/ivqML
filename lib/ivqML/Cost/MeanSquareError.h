// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__MeanSquareError__h__
#define __ivqML__Cost__MeanSquareError__h__

#include <ivqML/Cost/Base.h>

namespace ivqML
{
  namespace Cost
  {
    /**
     */
    template< class _TModel >
    class MeanSquareError
      : public ivqML::Cost::Base< _TModel >
    {
    public:
      using Self       = MeanSquareError;
      using Superclass = ivqML::Cost::Base< _TModel >;
      using TModel     = typename Superclass::TModel;
      using TScl       = typename Superclass::TScl;
      using TNat       = typename Superclass::TNat;
      using TMat       = typename Superclass::TMat;
      using TCol       = typename Superclass::TCol;
      using TRow       = typename Superclass::TRow;
      using TMatMap    = typename Superclass::TMatMap;
      using TColMap    = typename Superclass::TColMap;
      using TRowMap    = typename Superclass::TRowMap;
      using TMatCMap   = typename Superclass::TMatCMap;
      using TColCMap   = typename Superclass::TColCMap;
      using TRowCMap   = typename Superclass::TRowCMap;

    public:
      MeanSquareError( );
      virtual ~MeanSquareError( ) = default;

      virtual TScl operator()(
        const TModel& model, TScl* G = nullptr
        ) const override;

    protected:
      mutable TMat m_Z;
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/MeanSquareError.hxx>

#endif // __ivqML__Cost__MeanSquareError__h__

// eof - $RCSfile$
