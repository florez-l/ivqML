// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Logistic__h__
#define __ivqML__Model__Logistic__h__

#include <ivqML/Model/Linear.h>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _S >
    class Logistic
      : public ivqML::Model::Linear< _S >
    {
    public:
      using Self = Logistic;
      using Superclass = ivqML::Model::Linear< _S >;

      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;

    public:
      Logistic( const TNatural& n = 0 );
      virtual ~Logistic( ) = default;

      template< class _X >
      auto threshold( const Eigen::EigenBase< _X >& iX ) const
        {
          return(
            ( this->evaluate( iX ).array( ) >= TScalar( 0.5 ) )
            .template cast< TScalar >( )
            );
        }

    protected:
      virtual void _evaluate( const TNatural& m ) const override;
    };
  } // end namespace
} // end namespace

//// #include <ivqML/Model/Logistic.hxx>

#endif // __ivqML__Model__Logistic__h__

// eof - $RCSfile$
