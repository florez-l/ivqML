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
  using TActivation = std::function< TColVector( const TColVector& ) >;

public:
  Layer( );
  Layer( unsigned int i_size, unsigned int o_size, const TActivation& f );
  Layer( const TMatrix& w, const TColVector& b, const TActivation& f );
  Layer( const Self& other );
  virtual ~Layer( ) = default;
  Self& operator=( const Self& other );

  unsigned int input_size( ) const;
  unsigned int output_size( ) const;

  TMatrix& W( );
  const TMatrix& W( ) const;

  TColVector& B( );
  const TColVector& B( ) const;

  TActivation& sigma( );
  const TActivation& sigma( ) const;

  void init( bool randomly = true );

  TColVector operator()( const TColVector& z ) const;

protected:
  void _ReadFrom( std::istream& i );
  void _CopyTo( std::ostream& o ) const;

protected:
  TMatrix     m_W;
  TColVector  m_B;
  TActivation m_sigma;

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
