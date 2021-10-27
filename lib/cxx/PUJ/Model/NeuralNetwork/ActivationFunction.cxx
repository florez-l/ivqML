// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__NeuralNetwork__ActivationFunction__h__
#define __PUJ__NeuralNetwork__ActivationFunction__h__

#include <PUJ/Traits.h>

// -------------------------------------------------------------------------
#define PUJ_NN_ActivationFunction_Base( _n_ ) \
      template< class _TScalar, class _TTraits > \
      typename PUJ::Model::NeuralNetwork::_n_< _TScalar >::TMatrix \
      PUJ::Model::NeuralNetwork::_n_< _TScalar >:: \
      operator()( const TMatrix& z )

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( ArcTan )
{
}
      
// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( BinaryStep )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( ELU )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( Identity )
{
  return( z );
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( LeakyReLU )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( Logistic )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( RandomReLU )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( ReLU )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( SoftMax )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( SoftMin )
{
}

// -------------------------------------------------------------------------
PUJ_NN_ActivationFunction_Base( Tanh )
{
}

// eof - $RCSfile$
