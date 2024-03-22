// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__BinaryCrossEntropy__h__
#define __ivqML__Cost__BinaryCrossEntropy__h__

#include <ivqML/Cost/Base.h>

namespace ivqML
{
  namespace Cost
  {
    /**
     */
    template< class _TModel >
    class BinaryCrossEntropy
      : public ivqML::Cost::Base< _TModel >
    {
    public:
      using Self       = BinaryCrossEntropy;
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
      BinaryCrossEntropy( );
      virtual ~BinaryCrossEntropy( ) = default;

      virtual TScl operator()(
        const TModel& model, TScl* G = nullptr
        ) const override;
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/BinaryCrossEntropy.hxx>

#endif // __ivqML__Cost__BinaryCrossEntropy__h__

// eof - $RCSfile$
