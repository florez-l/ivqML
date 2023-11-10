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
    template< class _M, class _X = typename _M::TMatrix, class _Y = typename _M::TMatrix >
    class BinaryCrossEntropy
      : public ivqML::Cost::Base< _M, _X, _Y >
    {
    public:
      using Self = BinaryCrossEntropy;
      using Superclass = ivqML::Cost::Base< _M, _X, _Y >;

      ivqML_Cost_Typedefs;

    public:
      BinaryCrossEntropy( const _M& m, const TX& iX, const TY& iY );
      virtual ~BinaryCrossEntropy( ) override = default;

      virtual TResult operator()( const TNatural& b = 0 ) override;

    protected:
      TScalar m_Ym;
      TMatrix m_Xy;
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/BinaryCrossEntropy.hxx>

#endif // __ivqML__Cost__BinaryCrossEntropy__h__

// eof - $RCSfile$
