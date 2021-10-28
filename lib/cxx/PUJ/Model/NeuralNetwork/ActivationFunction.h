// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__NeuralNetwork__ActivationFunction__h__
#define __PUJ__NeuralNetwork__ActivationFunction__h__

#include <PUJ/Traits.h>

// -------------------------------------------------------------------------
#define PUJ_NN_ActivationFunction( _n_ )                                \
  template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >  \
  class PUJ_ML_EXPORT _n_                                               \
    : public ActivationFunction< _TScalar, _TTraits >                   \
  {                                                                     \
  public:                                                               \
    using Superclass = ActivationFunction< _TScalar, _TTraits >;        \
    PUJ_TraitsMacro( _n_ );                                             \
  public:                                                               \
    _n_( ) : Superclass( ) { }                                          \
    virtual ~_n_( ) = default;                                          \
    virtual TMatrix operator()( const TMatrix& z ) override;            \
    virtual TMatrix operator[]( const TMatrix& z ) override;            \
  }

// -------------------------------------------------------------------------
#define PUJ_NN_ActivationFunction_Parameter( _n_, _a_, _v_ )            \
  template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >  \
  class PUJ_ML_EXPORT _n_                                               \
    : public ActivationFunction< _TScalar, _TTraits >                   \
  {                                                                     \
  public:                                                               \
    using Superclass = ActivationFunction< _TScalar, _TTraits >;        \
    PUJ_TraitsMacro( _n_ );                                             \
  public:                                                               \
    _n_( ) : Superclass( ) { }                                          \
    virtual ~_n_( ) = default;                                          \
    void Set##_a_( const TScalar& v ) { this->m_##_a_ = v; }            \
    const TScalar& Get##_a_( ) const { return( this->m_##_a_ ); }       \
    virtual TMatrix operator()( const TMatrix& z ) override;            \
    virtual TMatrix operator[]( const TMatrix& z ) override;            \
  protected:                                                            \
    TScalar m_##_a_ { _v_ };                                            \
  }

namespace PUJ
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
      class PUJ_ML_EXPORT ActivationFunction
      {
      public:
        PUJ_TraitsMacro( ActivationFunction );

      public:
        ActivationFunction( ) { }
        virtual ~ActivationFunction( ) = default;

        virtual TMatrix operator()( const TMatrix& z ) = 0;
        virtual TMatrix operator[]( const TMatrix& z ) = 0;
      };

      // -- Basic activation functions
      PUJ_NN_ActivationFunction( ArcTan );
      PUJ_NN_ActivationFunction( BinaryStep );
      PUJ_NN_ActivationFunction( Identity );
      PUJ_NN_ActivationFunction( LeakyReLU );
      PUJ_NN_ActivationFunction( Logistic );
      PUJ_NN_ActivationFunction( ReLU );
      PUJ_NN_ActivationFunction( SoftMax );
      PUJ_NN_ActivationFunction( SoftPlus );
      PUJ_NN_ActivationFunction( Tanh );

      // -- Functions with parameters
      PUJ_NN_ActivationFunction_Parameter( ELU, Alpha, 1e-1 );
      PUJ_NN_ActivationFunction_Parameter( RandomReLU, Alpha, 1e-1 );

    } // end namespace
  } // end namespace
} // end namespace

#endif // __PUJ__NeuralNetwork__ActivationFunction__h__

// eof - $RCSfile$
