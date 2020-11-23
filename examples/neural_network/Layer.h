// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Layer__h__
#define __PUJ_ML__Layer__h__

#include "ActivationFunctions.h"

// -------------------------------------------------------------------------
#define _GetMacro_( _n_, _a_, _t_ )                                     \
  virtual _t_&       _n_( )       { return( this->m_##_a_ ); }          \
  virtual const _t_& _n_( ) const { return( this->m_##_a_ ); }

// -------------------------------------------------------------------------
#define _GetPointerMacro_( _n_, _a_, _t_ )                              \
  virtual _t_*       _n_( )       { return( this->m_##_a_ ); }          \
  virtual const _t_* _n_( ) const { return( this->m_##_a_ ); }

/**
 */
template< class _TScl >
class Layer
{
public:
  using Self        = Layer;
  using TFactory    = ActivationFunctions::Factory< _TScl >;
  using TActivation = typename TFactory::TFunction;
  using TScalar     = typename TActivation::TScalar;
  using TColVector  = typename TActivation::TColVector;
  using TRowVector  = typename TActivation::TRowVector;
  using TMatrix     = typename TActivation::TMatrix;

public:
  _GetMacro_( weights, W, TMatrix );
  _GetMacro_( biases, B, TColVector );
  _GetPointerMacro_( sigma, S, TActivation );

public:
  Layer( );
  Layer( unsigned int i_size, unsigned int o_size, const std::string& f );
  Layer( const TMatrix& w, const TColVector& b, const std::string& f );
  Layer( const Self& o );
  virtual ~Layer( );
  Self& operator=( const Self& o );

  unsigned int input_size( ) const;
  unsigned int output_size( ) const;

  void init( );
  TScalar regularization( ) const;

  TMatrix f( const TMatrix& x ) const;
  TMatrix f( const TMatrix& x, TMatrix& z ) const;

protected:
  void _read_from( std::istream& i );
  void _copy_to( std::ostream& o ) const;

protected:
  TMatrix      m_W;
  TColVector   m_B;
  TActivation* m_S;

public:
  ///!
  friend std::istream& operator>>( std::istream& i, Self& l )
    {
      l._read_from( i );
      return( i );
    }

  ///!
  friend std::ostream& operator<<( std::ostream& o, const Self& l )
    {
      l._copy_to( o );
      return( o );
    }
};

#endif // __PUJ_ML__Layer__h__

// eof - $RCSfile$
