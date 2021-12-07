// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Perceptron__h__
#define __PUJ_ML__Model__Perceptron__h__

#include <PUJ_ML/Model/Linear.h>

namespace PUJ_ML
{
  namespace Model
  {
    /**
     */
    template< class _T >
    class Perceptron
      : public PUJ_ML::Model::Linear< _T >
    {
    public:
      using Self       = Perceptron;
      using Superclass = PUJ_ML::Model::Linear< _T >;
      using TScalar    = typename Superclass::TScalar;
      using TMatrix    = typename Superclass::TMatrix;
      using TRow       = typename Superclass::TRow;
      using TColumn    = typename Superclass::TColumn;

    public:
      Perceptron( );
      Perceptron( const TMatrix& X, const TColumn& Y );
      virtual ~Perceptron( ) = default;

      virtual TColumn operator[]( const TMatrix& x ) override;
    };
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Model__Perceptron__h__

// eof - $RCSfile$
