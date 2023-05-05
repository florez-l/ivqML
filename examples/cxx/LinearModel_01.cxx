// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <vector>
#include <Eigen/Core>

namespace Model
{
  /**
   */
  template< class _R >
  class Base
  {
  public:
    using Self = Base;
    using TReal = _R;
    using TMatrix = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;
    using TCol = Eigen::Matrix< _R, Eigen::Dynamic, 1 >;
    using TRow = Eigen::Matrix< _R, 1, Eigen::Dynamic >;

  public:
    Base( const unsigned long long& n = 1 )
      {
        this->set_number_of_parameters( n );
      }
    virtual ~Base( ) = default;

    virtual void set_number_of_parameters( const unsigned long long& n )
      {
        this->m_P.resize( n, _R( 0 ) );
        this->m_P.shrink_to_fit( );
      }

    virtual _R& operator()( const unsigned long long& i )
      {
        static _R zero;
        if( i < this->m_P.size( ) )
          return( this->m_P[ i ] );
        else
        {
          zero = 0;
          return( zero );
        } // end if
      }
    virtual const _R& operator()( const unsigned long long& i ) const
      {
        static const _R zero = 0;
        if( i < this->m_P.size( ) )
          return( this->m_P[ i ] );
        else
          return( zero );
      }

    virtual _R& operator()(
      const unsigned long long& i,
      const unsigned long long& j
      )
      {
        return( this->operator()( i ) );
      }

    virtual const _R& operator()(
      const unsigned long long& i,
      const unsigned long long& j
      ) const
      {
        return( this->operator()( i ) );
      }

    template< class _Y, class _X >
    void evaluate(
      Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
      ) const
      {
      }

    template< class _Y, class _X >
    void threshold(
      Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
      ) const
      {
        this->evaluate( Y, X );
      }

  protected:
    virtual void _to_stream( std::ostream& o ) const
      {
        o << this->m_P.size( );
        for( const _R& v: this->m_P )
          o << " " << v;
      }

  protected:
    std::vector< _R > m_P;

  public:
    friend std::ostream& operator<<( std::ostream& o, const Self& m )
      {
        m._to_stream( o );
        return( o );
      }
  };
} // end namespace

namespace Model
{
  /**
   */
  template< class _R >
  class Linear
    : public Model::Base< _R >
  {
  public:
    using Self = Linear;
    using Superclass = Model::Base< _R >;
    using TReal = typename Superclass::TReal;
    using TMatrix = typename Superclass::TMatrix;
    using TCol = typename Superclass::TCol;
    using TRow = typename Superclass::TRow;

  public:
    Linear( const unsigned long long& n = 1 )
      : Superclass( n )
      {
      }
    virtual ~Linear( )
      {
        if( this->m_T != nullptr )
          delete this->m_T;
      }

    unsigned long long number_of_parameters( ) const
      {
        return( this->m_P.size( ) );
      }

    virtual void set_number_of_parameters(
      const unsigned long long& n
      ) override
      {
        this->Superclass::set_number_of_parameters( n + 1 );
        this->m_T =
          new Eigen::Map< TCol >(
            this->m_P.data( ) + 1, this->m_P.size( ) - 1, 1
            );
      }

    template< class _Y, class _X >
    void evaluate(
      Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
      ) const
      {
        Y.derived( ) = ( ( X.derived( ).template cast< _R >( ) * *( this->m_T ) ).array( ) + this->m_P[ 0 ] ).template cast< typename _Y::Scalar >( );
      }

  protected:
    Eigen::Map< TCol >* m_T { nullptr };
  };
} // end namespace

int main( int argc, char** argv )
{
  Model::Linear< double > model;
  model.set_number_of_parameters( 1 );
  model( 0 ) = 1;
  model( 1 ) = 3;

  decltype( model )::TMatrix X( 4, model.number_of_parameters( ) - 1 );
  X << 1, 2, 3, 4;
  decltype( model )::TCol Y;
  model.evaluate( Y, X );

  std::cout << "===============" << std::endl;
  std::cout << model << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << Y << std::endl;
  std::cout << "===============" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
