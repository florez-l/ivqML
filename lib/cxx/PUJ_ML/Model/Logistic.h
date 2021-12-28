// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Logistic__h__
#define __PUJ_ML__Model__Logistic__h__

#include <PUJ_ML/Model/Linear.h>

namespace PUJ_ML
{
  namespace Model
  {
    /**
     */
    template< class _T >
    class Logistic
      : public PUJ_ML::Model::Linear< _T >
    {
    public:
      using Self       = Logistic;
      using Superclass = PUJ_ML::Model::Linear< _T >;
      using TScalar    = typename Superclass::TScalar;
      using TMatrix    = typename Superclass::TMatrix;
      using TRow       = typename Superclass::TRow;
      using TCol       = typename Superclass::TCol;

    public:
      /**
       */
      class Cost
        : public Superclass::Cost
      {
      public:
        Cost( Superclass* model, const TMatrix& X, const TMatrix& Y );
        virtual ~Cost( ) = default;

        virtual _T operator()( _T* g = nullptr ) const override;
      };

    public:
      Logistic( );
      Logistic( const TMatrix& X, const TCol& Y );
      virtual ~Logistic( ) = default;

      virtual TMatrix operator()( const TMatrix& x ) override;
      virtual TMatrix operator[]( const TMatrix& x ) override;
    };
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Model__Logistic__h__

// eof - $RCSfile$
