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
      using TScalar    = typename Superclass::TScalar;
      using TNatural   = typename Superclass::TNatural;
      using TMatrix    = typename Superclass::TMatrix;
      using TColumn    = typename Superclass::TColumn;
      using TRow       = typename Superclass::TRow;

    public:
      MeanSquareError( _TModel& m );
      virtual ~MeanSquareError( ) = default;

      virtual TScalar operator()( TScalar* G = nullptr ) const override;
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/MeanSquareError.hxx>

#endif // __ivqML__Cost__MeanSquareError__h__

// eof - $RCSfile$
