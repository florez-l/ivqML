// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Layer__h__
#define __PUJ_ML__Layer__h__

#include <functional>
#include <Eigen/Core>

template< class _TScalar >
class Layer
{
public:
  using Self    = Layer;
  using TScalar = _TScalar;

  using TMatrix     = Eigen::Matrix< TScalar, Eigen::Dynamic, Eigen::Dynamic >;
  using TRowVector  = Eigen::Matrix< TScalar, 1, Eigen::Dynamic >;
  using TColVector  = Eigen::Matrix< TScalar, Eigen::Dynamic, 1 >;
  using TActivation = std::function< TMatrix( const TMatrix&, bool ) >;

public:
  Layer( );
  Layer( unsigned int i_size, unsigned int o_size, const TActivation& f );
  Layer( const TMatrix& w, const TColVector& b, const TActivation& f );
  Layer( const Self& other );
  virtual ~Layer( ) = default;
  Self& operator=( const Self& other );

  unsigned int input_size( ) const;
  unsigned int output_size( ) const;

  TMatrix& weights( );
  const TMatrix& weights( ) const;

  TColVector& biases( );
  const TColVector& biases( ) const;

  TActivation& sigma( );
  const TActivation& sigma( ) const;

  void init( bool randomly = true );

  TColVector linear_fwd( const TColVector& x ) const;
  TColVector sigma_fwd( const TColVector& z ) const;
  TColVector delta_bck( const TColVector& d, const TColVector& z ) const;
  TMatrix operator()( const TMatrix& X ) const;

  TScalar regularization( ) const;

protected:
  void _ReadFrom( std::istream& i );
  void _CopyTo( std::ostream& o ) const;

protected:
  TMatrix     m_W;
  TColVector  m_B;
  TActivation m_S;

public:
  ///!
  friend std::istream& operator>>( std::istream& i, Self& l )
    {
      l._ReadFrom( i );
      return( i );
    }

  ///!
  friend std::ostream& operator<<( std::ostream& o, const Self& l )
    {
      l._CopyTo( o );
      return( o );
    }
};

#endif // __PUJ_ML__Layer__h__

// eof - $RCSfile$
